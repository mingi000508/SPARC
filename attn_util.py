import torch
from types import MethodType
from functools import partial
from typing import Dict, Any, Optional, Tuple
import torch.nn.functional as F
import torch.nn as nn
import logging
from dataclasses import dataclass
import math
from sklearn.cluster import KMeans
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    rotate_half,
    repeat_kv,
)
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)


class SelectedIndexBuffer:
    def __init__(self):
        self.indices1 = (
            []
        )  # Buffer to store selected token indices from the current token generation step
        self.indices2 = (
            []
        )  # Buffer to store selected token indices from the previous token generation step
        self.input_len = 0  # Length of the input sequence
        self.num_image_patches = None

    def update_indices1(self, indices, image_token_index):
        """
        Updates indices1 with the selected token indices from a specific layer (e.g., the 20th layer).
        This is called during the current token generation to store new selections.
        The image_token_index offset is applied to adjust for the position of image tokens.
        """
        indices = indices + image_token_index
        indices = indices.squeeze(dim=-1)
        self.indices1 = indices

    def update_indices2(self):
        """
        Copies indices1 to indices2.
        Called at the beginning of a new token generation step so that calibration can use
        the indices selected from the previous step.
        """
        self.indices2 = self.indices1

    def update_input_len(self, len):
        self.input_len = len

    def reset(self):
        self.indices1 = []
        self.indices2 = []
        self.input_len = 0
        self.num_image_patches = None

    def calibrate(self, value, alpha):
        """
        Applies calibration to token representations using indices from the previous step (indices2).
        The calibration scales selected token positions by a factor alpha.
        """
        if len(self.indices2) > 0:
            value[:, :, self.indices2] *= alpha

    def update_patch_num(self, num_image_patches):
        self.num_image_patches = num_image_patches


# llama attention
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    image_token_index: Optional[int] = 35,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    tau: Optional[float] = 2,
    selected: Optional[bool] = False,
    se_layers: Optional[Tuple[int, int]] = None,
    indices_buffer: Optional[SelectedIndexBuffer] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if self.layer_idx == 0:
        indices_buffer.update_indices2()

    if self.layer_idx >= se_layers[0] and self.layer_idx <= se_layers[1]:
        if len(indices_buffer.indices2) > 0:
            indices_buffer.calibrate(past_key_value.value_cache[self.layer_idx], alpha)

    if len(past_key_value.key_cache) > self.layer_idx:
        # generation
        gen_new_token = True
    else:
        # pre-filling
        gen_new_token = False

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )

    if gen_new_token == False and self.layer_idx == 0:
        indices_buffer.update_patch_num(
            attn_weights.shape[-1] - indices_buffer.input_len
        )

    image_attention = attn_weights[
        :,
        :,
        -1,
        image_token_index : image_token_index + indices_buffer.num_image_patches,
    ].mean(dim=1)

    if gen_new_token:
        if selected:
            # mean by head dim
            ratio = (image_attention - self.image_attention) / self.image_attention
            ratio = ratio.squeeze(dim=0)
            indices = (ratio >= tau).nonzero()
            indices_buffer.update_indices1(indices, image_token_index)

    if not gen_new_token:
        self.image_attention = image_attention
    else:
        self.image_attention = (
            1 - beta
        ) * image_attention + beta * self.image_attention

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def add_custom_attention_layers(
    model,
    lm_model="llama",
    alpha=1,
    beta=0,
    tau=2,
    selected_layer=20,
    se_layers=(0, 31),
    indices_buffer=None,
):

    for i, layer in enumerate(model.model.layers):
        selected = True if selected_layer == i else False
        forward_ = partial(
            forward,
            alpha=alpha,
            beta=beta,
            tau=tau,
            selected=selected,
            se_layers=se_layers,
            indices_buffer=indices_buffer,
        )
        layer.self_attn.forward = MethodType(forward_, layer.self_attn)
