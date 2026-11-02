"""Synthetic Llama-named checkpoints, LoRA adapters and direction files for lineage tests.

Everything is built in-process with torch and safetensors; nothing is downloaded. Import this
module only after ``pytest.importorskip("torch")``.
"""

import json

import numpy as np
import torch
from safetensors import torch as st

from changegate.checks.lineage_tensors import block_of, writer_axis
from changegate.config import Envelope

L, D, FF, V = 4, 16, 32, 48
ENV = Envelope(warn=0.2, fail=0.5, direction="higher_is_worse")
CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": D,
    "num_hidden_layers": L,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": FF,
    "vocab_size": V,
    "torch_dtype": "float32",
    "transformers_version": "4.57.6",
}
VOCAB = {f"t{i}": i for i in range(V - 2)}
ADDED = [{"id": V - 2, "content": "<|im_start|>"}, {"id": V - 1, "content": "<|im_end|>"}]


def weights(seed, layers=L, d=D, vocab=V):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g) * 0.1

    t = {
        "model.embed_tokens.weight": rnd(vocab, d),
        "model.norm.weight": torch.ones(d),
        "lm_head.weight": rnd(vocab, d),
    }
    for i in range(layers):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = torch.ones(d)
        t[p + "post_attention_layernorm.weight"] = torch.ones(d)
        for n in "qkvo":
            t[p + f"self_attn.{n}_proj.weight"] = rnd(d, d)
        t[p + "mlp.gate_proj.weight"] = rnd(FF, d)
        t[p + "mlp.up_proj.weight"] = rnd(FF, d)
        t[p + "mlp.down_proj.weight"] = rnd(d, FF)
    return t


def unit(seed, d=D):
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def abliterate(t, r_for, blocks=None, embed=True):
    """Orthogonalize residual writers against r (Arditi-style weight edit)."""
    out = dict(t)
    for name, w in t.items():
        block = block_of(name)
        if writer_axis(name) == 0 and (blocks is None or block in blocks):
            r = r_for(block)
            out[name] = w - torch.outer(r, r @ w)
        elif writer_axis(name) == 1 and embed:
            r = r_for(None)
            out[name] = w - torch.outer(w @ r, r)
    return out


def nudge(t, scale=0.01, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {
        k: v + scale * v.std() * torch.randn(v.shape, generator=g) if v.ndim == 2 else v
        for k, v in t.items()
    }


def save(
    path,
    tensors,
    *,
    config=CONFIG,
    vocab=VOCAB,
    added=ADDED,
    template="{{ m }}",
    dtype=None,
    shards=1,
):
    path.mkdir(parents=True)
    tensors = {k: (v.to(dtype) if dtype else v).contiguous() for k, v in tensors.items()}
    names = sorted(tensors)
    if shards == 1:
        st.save_file(tensors, str(path / "model.safetensors"))
    else:
        weight_map = {}
        for i in range(shards):
            shard = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
            part = names[i::shards]
            st.save_file({n: tensors[n] for n in part}, str(path / shard))
            weight_map.update(dict.fromkeys(part, shard))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    if config is not None:
        (path / "config.json").write_text(json.dumps(config))
    if vocab is not None:
        tok = {"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added}
        (path / "tokenizer.json").write_text(json.dumps(tok))
        tok_cfg = {"chat_template": template, "eos_token": {"content": "<|im_end|>"}}
        (path / "tokenizer_config.json").write_text(json.dumps(tok_cfg))
    return path


def lora(path, pairs, *, alpha=1, r=1, extra=None, config_extra=None, save_weights=True):
    path.mkdir(parents=True)
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": alpha, "base_model_name_or_path": "tiny/base"}
    cfg.update(config_extra or {})
    (path / "adapter_config.json").write_text(json.dumps(cfg))
    tensors = {}
    for module, (a, b) in pairs.items():
        tensors[f"base_model.model.{module}.lora_A.weight"] = a.contiguous()
        tensors[f"base_model.model.{module}.lora_B.weight"] = b.contiguous()
    tensors.update(extra or {})
    if save_weights:
        st.save_file(tensors, str(path / "adapter_model.safetensors"))
    return path


def npy(path, array):
    np.save(path, np.asarray(array, dtype=np.float32))
    return path
