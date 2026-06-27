"""Profile a Figure-2-style runtime breakdown for local fine-tuning methods.

The SparseLoRA paper's Figure 2 sweeps sequence length.  This script keeps a
single sequence length fixed and compares methods instead.  It writes both raw
JSON and a stacked-bar plot under ``checkpoints/runtime_breakdown`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparselora import SparseLoRAConfig, apply_sparselora

from EfficientRED.models import ActivationLLama


CATEGORIES = ["Layer Norm", "RoPE", "LoRA Branch", "QKVO Proj", "Attention", "FFN", "Other"]
DEFAULT_METHODS = ("lora", "red", "sparselora")
DEFAULT_N_WARMUP = 5
DEFAULT_N_ITERS = 20
DEFAULT_PROFILE_LAST_N = 5
DEFAULT_OUTPUT_NAME = "runtime_breakdown_512"
DEFAULT_PLOT_UNITS = "ms"


@dataclass
class MethodResult:
    method: str
    max_seq_length: int
    batch_size: int
    warmup_steps: int
    profile_steps: int
    total_step_ms: float
    categorized_ms: dict[str, float]
    percent: dict[str, float]
    forward_ms: float
    backward_ms: float
    total_ms: float


def _parse_val(v: str):
    try:
        from ast import literal_eval

        return literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def _parse_kv_args(spec: str) -> dict:
    if not spec:
        return {}
    return {k: _parse_val(v) for k, v in (item.split("=", 1) for item in spec.split(","))}


def tokenize_and_mask(tokenizer, max_len, data_point):
    instruction = data_point["instruction"]
    response = data_point.get("output", data_point.get("answer", ""))
    full = tokenizer(f"{instruction}\n{response}", truncation=True, max_length=max_len)
    if full["input_ids"][-1] != tokenizer.eos_token_id and len(full["input_ids"]) < max_len:
        full["input_ids"].append(tokenizer.eos_token_id)
    prefix = tokenizer(f"{instruction}\n", truncation=True, max_length=max_len)
    labels = list(full["input_ids"])
    labels[: len(prefix["input_ids"])] = [-100] * len(prefix["input_ids"])
    return {"input_ids": full["input_ids"], "labels": labels}


def build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset_and_collator(args, tokenizer):
    from functools import partial

    dataset = load_dataset("json", data_files=args.dataset)["train"]
    dataset = dataset.map(partial(tokenize_and_mask, tokenizer, args.max_seq_length))
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=args.max_seq_length,
        return_tensors="pt",
        padding=True,
    )
    return dataset, data_collator


def build_model(args, method: str):
    if method == "sparselora":
        from liger_kernel.transformers import apply_liger_kernel_to_llama

        apply_liger_kernel_to_llama(
            rope=True,
            swiglu=False,
            cross_entropy=True,
            fused_linear_cross_entropy=False,
            rms_norm=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )

    if method == "red":
        model = ActivationLLama(model)

    elif method == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    elif method == "sparselora":
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        config = SparseLoRAConfig.from_pretrained(**_parse_kv_args(args.sparselora))
        model = apply_sparselora(model, config)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


def load_batch(dataset, data_collator, batch_size, device):
    keep_keys = {"input_ids", "attention_mask", "labels"}
    features = [
        {k: v for k, v in dataset[i % len(dataset)].items() if k in keep_keys}
        for i in range(batch_size)
    ]
    batch = data_collator(features)
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def module_category(name: str, module: nn.Module) -> str | None:
    cls = type(module).__name__.lower()
    lname = name.lower()
    if "lora" in lname:
        return "LoRA Branch"
    if cls.endswith("rmsnorm") or "layernorm" in cls:
        return "Layer Norm"
    if cls in {"llamamlp", "sparsellamamlp"} or "mlp" in name or "lm_head" in name:
        return "FFN"
    if cls in {"llamarotaryembedding", "embedding"}:
        return "RoPE"
    if cls in {"gqaattentionpredictor"}:
        return "Attention"
    qkvo_names = (".q_proj", ".k_proj", ".v_proj", ".o_proj")
    if name.endswith(qkvo_names) or any(part in name for part in (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.")):
        if cls in {"linear", "sparselinear"} or "peft" in type(module).__module__:
            return "QKVO Proj"
    return None


def _wrap_forward(module: nn.Module, category: str) -> Callable:
    original = module.forward

    def wrapped(*args, **kwargs):
        with record_function(f"runtime_breakdown::{category}"):
            return original(*args, **kwargs)

    module.forward = wrapped
    return original


def _register_backward_range(module: nn.Module, category: str, contexts: dict[int, list]):
    def pre_hook(mod, grad_output):
        ctx = record_function(f"runtime_breakdown_backward::{category}")
        ctx.__enter__()
        contexts[id(mod)].append(ctx)

    def post_hook(mod, grad_input, grad_output):
        stack = contexts.get(id(mod))
        if stack:
            stack.pop().__exit__(None, None, None)

    return module.register_full_backward_pre_hook(pre_hook), module.register_full_backward_hook(post_hook)


@contextmanager
def instrument_model(model: nn.Module):
    originals = []
    handles = []
    backward_contexts = defaultdict(list)
    for name, module in model.named_modules():
        if not next(module.children(), None):
            category = module_category(name, module)
            if category is not None:
                originals.append((module, _wrap_forward(module, category)))
                handles.extend(_register_backward_range(module, category, backward_contexts))

    rope_originals = []
    attention_originals = []
    try:
        import transformers.models.llama.modeling_llama as llama_modeling

        rope_originals.append((llama_modeling, "apply_rotary_pos_emb", llama_modeling.apply_rotary_pos_emb))

        def wrapped_rope(*args, **kwargs):
            with record_function("runtime_breakdown::RoPE"):
                return rope_originals[0][2](*args, **kwargs)

        llama_modeling.apply_rotary_pos_emb = wrapped_rope
    except (AttributeError, ImportError):
        pass

    try:
        import sparselora.modules.llama as sparse_llama

        sparse_original = sparse_llama.apply_rotary_pos_emb
        rope_originals.append((sparse_llama, "apply_rotary_pos_emb", sparse_original))

        def wrapped_sparse_rope(*args, **kwargs):
            with record_function("runtime_breakdown::RoPE"):
                return sparse_original(*args, **kwargs)

        sparse_llama.apply_rotary_pos_emb = wrapped_sparse_rope
    except (AttributeError, ImportError):
        pass

    try:
        import torch.nn.functional as F

        attention_originals.append((F, "scaled_dot_product_attention", F.scaled_dot_product_attention))

        def wrapped_sdpa(*args, **kwargs):
            with record_function("runtime_breakdown::Attention"):
                return attention_originals[0][2](*args, **kwargs)

        F.scaled_dot_product_attention = wrapped_sdpa
    except (AttributeError, ImportError):
        pass

    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original
        for handle in handles:
            handle.remove()
        for owner, attr, original in rope_originals:
            setattr(owner, attr, original)
        for owner, attr, original in attention_originals:
            setattr(owner, attr, original)


def run_step(model: nn.Module, batch: dict) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss if hasattr(out, "loss") else out[0]
    loss.backward()
    return loss.detach()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_steps(model: nn.Module, batch: dict, steps: int) -> float:
    synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        run_step(model, batch)
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / steps


def profiler_self_time_us(item) -> float:
    """Return the best available self time from a profiler aggregate event."""
    for attr in ("self_cuda_time_total", "self_device_time_total", "self_cpu_time_total"):
        if hasattr(item, attr):
            return getattr(item, attr)
    raise AttributeError(f"No self-time field found on profiler event: {item.key}")


def profile_method(method: str, args, training_args, dataset, data_collator) -> MethodResult:
    device = training_args.device
    model = build_model(args, method)
    model.to(device)
    model.train()
    batch = load_batch(
        dataset, data_collator, training_args.per_device_train_batch_size, device
    )

    warmup_steps = DEFAULT_N_WARMUP
    profile_steps = DEFAULT_PROFILE_LAST_N

    for _ in range(warmup_steps):
        run_step(model, batch)
    total_step_ms = time_steps(model, batch, profile_steps)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with instrument_model(model):
        with profile(activities=activities) as prof:
            for _ in range(profile_steps):
                run_step(model, batch)

    categorized = defaultdict(float)
    forward_ms, backward_ms = 0, 0
    for item in prof.key_averages():
        if item.key.startswith("runtime_breakdown::"):
            category = item.key.split("::", 1)[1]
            categorized[category] += profiler_self_time_us(item) / 1000.0 / profile_steps
            forward_ms += profiler_self_time_us(item) / 1000.0 / profile_steps
        elif item.key.startswith("runtime_breakdown_backward::"):
            category = item.key.split("::", 1)[1]
            categorized[category] += profiler_self_time_us(item) / 1000.0 / profile_steps
            backward_ms += profiler_self_time_us(item) / 1000.0 / profile_steps

    known_ms = sum(categorized.values())
    categorized["Other"] = 0.0
    total_for_plot = sum(categorized.values()) or 1.0
    percent = {category: 100.0 * categorized.get(category, 0.0) / total_for_plot for category in CATEGORIES}

    result = MethodResult(
        method=method,
        max_seq_length=args.max_seq_length,
        batch_size=training_args.per_device_train_batch_size,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        total_step_ms=total_step_ms,
        categorized_ms={category: categorized.get(category, 0.0) for category in CATEGORIES},
        percent=percent,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        total_ms=forward_ms + backward_ms,
    )
    del model, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_json(results: list[MethodResult], out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump([asdict(result) for result in results], f, indent=2)
    return path


def _label_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance > 155 else "white"


def write_plot(results: list[MethodResult], out_dir: str, name: str, plot_units: str) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    methods = [result.method for result in results]
    colors = {
        "Layer Norm": "#4c78a8",
        "RoPE": "#f58518",
        "LoRA Branch": "#54a24b",
        "QKVO Proj": "#e45756",
        "Attention": "#72b7b2",
        "FFN": "#b279a2",
        "Other": "#8f8f8f",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bottoms = [0.0] * len(results)
    for category in CATEGORIES:
        if plot_units == "ms":
            values = [result.categorized_ms.get(category, 0.0) for result in results]
        else:
            values = [result.percent.get(category, 0.0) for result in results]

        bars = ax.bar(methods, values, bottom=bottoms, label=category, color=colors[category])
        for bar, value, bottom in zip(bars, values, bottoms):
            if value <= 0:
                continue
            label = f"{value:.1f} ms" if plot_units == "ms" else f"{value:.1f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + value / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=_label_color(colors[category]),
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    seq_len = results[0].max_seq_length if results else "?"
    title = f"Runtime breakdown at sequence length {seq_len}"
    ax.set_title(title)
    if plot_units == "ms":
        ax.set_ylabel("Runtime (ms)")
        ax.set_ylim(0, max(bottoms) * 1.08 if bottoms else 1)
    else:
        ax.set_ylabel("Runtime (%)")
        ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    del fig, ax, path, seq_len

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_fb.png")
    methods = [result.method for result in results]
    forward = [result.forward_ms for result in results]
    backward = [result.backward_ms for result in results]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(methods, forward, label="Forward", color="#4c78a8")
    ax.bar(methods, backward, bottom=forward, label="Backward", color="#f58518")

    for idx, result in enumerate(results):
        ax.text(idx, result.forward_ms / 2, f"{result.forward_ms:.1f}", ha="center", va="center", color="white")
        ax.text(
            idx,
            result.forward_ms + result.backward_ms / 2,
            f"{result.backward_ms:.1f}",
            ha="center",
            va="center",
            color="black",
        )
        ax.text(idx, result.total_ms, f"{result.total_ms:.1f} ms", ha="center", va="bottom", fontsize=8)

    seq_len = results[0].max_seq_length if results else "?"
    ax.set_title(f"Forward/backward latency at sequence length {seq_len}")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max((result.total_ms for result in results), default=1.0) * 1.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def measure(args, training_args):
    methods = DEFAULT_METHODS
    
    tokenizer = build_tokenizer(args)
    dataset, data_collator = build_dataset_and_collator(args, tokenizer)
    results = [profile_method(method, args, training_args, dataset, data_collator) for method in methods]
    if not is_main_process():
        return
    json_path = write_json(results, training_args.output_dir, DEFAULT_OUTPUT_NAME)
    plot_path = write_plot(results, training_args.output_dir, DEFAULT_OUTPUT_NAME, plot_units=DEFAULT_PLOT_UNITS)
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    raise SystemExit("profile_runtime_breakdown.py is intended to be called from experiments/train.py via measure().")
