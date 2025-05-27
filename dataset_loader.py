import random
import os
import json
import math
import torch

from torch.utils.data import Dataset, DataLoader
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from LLaVA.llava.mm_utils import (
    tokenizer_image_token,
    process_images,
)
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        dataset_type,
        annotations,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
    ):
        self.dataset_type = dataset_type
        self.annotations = annotations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.annotations[index]

        if self.dataset_type == "iiw":
            image_file = line["image/key"]

            img_id = image_file
            image_file = image_file + ".jpg"

        elif self.dataset_type == "docci":
            image_file = line["image_file"]

            img_id = image_file

        elif self.dataset_type == "coco":
            image_file = line
            img_id = int(image_file.split(".jpg")[0][-6:])

        qs = "Please describe this image in detail."

        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[
            0
        ]

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids, image_tensor, image.size, img_id

    def __len__(self):
        return len(self.annotations)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, img_id = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, img_id


# DataLoader
def create_data_loader(
    dataset_type,
    annotations,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        dataset_type,
        annotations,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def prepare_dataloader(
    dataset_type,
    annotation_file,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    num_chunks=1,
    chunk_idx=0,
    seed=42,
):
    # Dataset type
    assert dataset_type in ["coco", "iiw", "docci"]

    if dataset_type == "coco":
        annotations = os.listdir(image_folder)
    else:
        annotations = [
            json.loads(q) for q in open(os.path.expanduser(annotation_file), "r")
        ]
        annotations = get_chunk(annotations, num_chunks, chunk_idx)

    if dataset_type in ["coco", "docci"]:
        rng = random.Random(seed)
        annotations = rng.sample(annotations, 500)

    data_loader = create_data_loader(
        dataset_type,
        annotations,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
    )

    return data_loader, annotations
