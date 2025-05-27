import argparse
import torch
import os
import json
from tqdm import tqdm


from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path

from attn_util import (
    add_custom_attention_layers,
    SelectedIndexBuffer,
)
from dataset_loader import prepare_dataloader


def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    buffer = SelectedIndexBuffer()

    if args.alpha != 1.0:
        add_custom_attention_layers(
            model=model,
            alpha=args.alpha,
            beta=args.beta,
            tau=args.tau,
            selected_layer=args.selected_layer,
            se_layers=(args.start_layer, args.end_layer),
            indices_buffer=buffer,
        )

    data_loader, annotations = prepare_dataloader(
        dataset_type=args.dataset_type,
        annotation_file=args.annotation_file,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        seed=args.seed,
    )

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ans_file = open(
        os.path.join(save_path, "{}.jsonl".format(args.experiment_name)), "w"
    )

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    for (input_ids, image_tensor, image_sizes, img_id), line in tqdm(
        zip(data_loader, annotations), total=len(annotations)
    ):
        input_ids = input_ids.to(device="cuda", non_blocking=True)

        img_save = {}
        img_save["image_id"] = img_id[0]

        buffer.update_input_len(len(input_ids[0]) - 1)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                ),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        buffer.reset()

        img_save["caption"] = outputs
        ans_file.write(json.dumps(img_save) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/home/mingi/experiments/LLaVA/data/eval/DOCCI/images_aar",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="/home/mingi/experiments/LLaVA/data/eval/imageinwords/IIW-400/data.jsonl",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/mingi/experiments/SPARC/results",
    )
    parser.add_argument("--dataset_type", type=str, default="iiw")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=2)
    parser.add_argument("--selected_layer", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)
