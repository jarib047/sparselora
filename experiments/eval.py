"""Evaluation script for SparseLoRA-trained models."""

import argparse
import json
import os
import re

import torch
from peft import AutoPeftModelForCausalLM
from tabulate import tabulate
from torch import distributed as dist
from tqdm import trange
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from EfficientRED.models import load_REDllama_model
from safetensors.torch import load_file
from datetime import datetime

MATH_DATASETS = {"gsm8k", "mawps", "svamp"}

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

ANSWER_PATTERNS = {
    "boolq": (r"true|false", ""),
    "piqa": (r"1|2", "solution"),
    "social_i_qa": (r"1|2|3|4|5", "answer"),
    "arc-challenge": (r"1|2|3|4|5", "answer"),
    "arc-easy": (r"1|2|3|4|5", "answer"),
    "openbookqa": (r"1|2|3|4|5", "answer"),
    "hellaswag": (r"1|2|3|4", "ending"),
    "winogrande": (r"1|2", "option"),
}


def extract_answer(response: str, dataset: str):
    response = response.strip().lower()
    if dataset in MATH_DATASETS:
        nums = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))
        return float(nums[-1]) if nums else float("inf")
    pattern, prefix = ANSWER_PATTERNS[dataset]
    m = re.findall(pattern, response)
    return prefix + m[0] if m else ""


def match(pred, target, dataset: str) -> bool:
    if dataset in MATH_DATASETS:
        return abs(float(target) - pred) <= 0.001
    return pred == target


def rank():
    return int(os.environ.get("RANK", 0))


def world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--debug_samples", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--red_path", required=False, default="")
    parser.add_argument("--peft", required=False, default="not_red")
    args = parser.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if args.peft == "red":
        model = load_REDllama_model(args.model_name_or_path)
        state_dict = load_file(args.red_path)
        model.load_state_dict(state_dict)
        model = model.to(local_rank) 
        model.base_model.generation_config.max_length = GenerationConfig().max_length  # default is usually 20
        model.base_model.generation_config.max_new_tokens = None
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            model_max_length=512,
            padding_side="left",
            use_fast=False,
        )

    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="sdpa",
            torch_dtype=dtype_map[args.dtype],
            device_map="auto",
        )
        model.generation_config.max_length = GenerationConfig().max_length  # default is usually 20
        model.generation_config.max_new_tokens = None
        tokenizer = AutoTokenizer.from_pretrained(
            model.peft_config["default"].base_model_name_or_path,
            model_max_length=512,
            padding_side="left",
            use_fast=False,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        renormalize_logits=True,
        remove_invalid_values=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    metrics = {}
    for ds in args.dataset.split("+"):
        with open(os.path.join("datasets", ds, "test.json")) as f:
            instances = json.load(f)[rank() :: world_size()]

        correct, total = 0, 0
        for k in trange(0, len(instances), args.batch_size, disable=rank() != 0, desc=ds):
            batch = instances[k : k + args.batch_size]
            # prompts = [PROMPT_TEMPLATE.format(instruction=b["instruction"]) for b in batch]
            prompts = [f"{b['instruction']}\n" for b in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.inference_mode():
                out_ids = model.generate(**inputs, generation_config=gen_cfg)
            # responses = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            # for resp, b in zip(responses, batch):
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = out_ids[:, prompt_len:]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for answer_text, b in zip(responses, batch):
                pred = extract_answer(answer_text, ds)
                if rank() == 0 and args.debug_samples > 0:
                    print(
                        json.dumps(
                            {
                                "dataset": ds,
                                "target": b["answer"],
                                "pred": pred,
                                "response": answer_text.strip(),
                            },
                            ensure_ascii=False,
                        )
                    )
                    args.debug_samples -= 1
                correct += match(pred, b["answer"], ds)
            total += len(batch)

        if distributed:
            gathered = [None] * world_size()
            dist.all_gather_object(gathered, (correct, total))
        else:
            gathered = [(correct, total)]
        metrics[ds] = sum(c for c, _ in gathered) / sum(t for _, t in gathered)

    if rank() == 0:
        print(tabulate(metrics.items(), headers=["Dataset", "Accuracy"], tablefmt="simple_outline"))
        if args.peft == "red":
            dir_path = os.path.dirname(args.red_path)
        else:
            dir_path = args.model_name_or_path
        out_path = os.path.join(dir_path, "metrics.json")
        if os.path.isdir(dir_path):
            if os.path.isfile(out_path):
                with open(out_path, "a") as f:
                    f.write(f"\n\nLatest result as of {datetime.now()}")
                    json.dump(metrics, f, indent=2)
            else:
                with open(out_path, "w") as f:
                    f.write(f"\n\nLatest result as of {datetime.now()}")
                    json.dump(metrics, f, indent=2)
            with open(os.path.join(dir_path, "eval_config.json"), "w") as f:
                json.dump(
                    {
                        "model_name_or_path": args.model_name_or_path,
                        "dataset": args.dataset,
                        "batch_size": args.batch_size,
                        "max_new_tokens": args.max_new_tokens,
                        "dtype": args.dtype,
                        "peft": args.peft,
                        "red_path": args.red_path,
                        "prompt_format": "instruction_newline",
                        "decode_generated_tokens_only": True,
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
