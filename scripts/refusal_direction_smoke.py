#!/usr/bin/env python3
"""Compute one model's refusal direction end to end and print sanity numbers beside it.

Loads one checkpoint (a local directory, or a Hugging Face repo id resolved through the hub
cache, downloading safetensors and tokenizer files when the cache misses and the hub is not
offline), reads residual-stream activations on the harmful and harmless prompt sets, takes the
difference-of-means direction (Arditi et al., arXiv:2406.11717) at one layer, saves it as a
.npy with metadata, and prints:

- the cosine of the direction with a seeded random unit vector, beside 1/sqrt(d), the scale
  of that cosine when nothing aligns;
- the split-half cosine between directions from even- and odd-indexed prompts;
- harmful/harmless d' along the direction; then the same three numbers for every layer.

It prints measurements only: no envelope is applied and no verdict is given. It appends
``smoke.start`` and ``smoke.end`` rows to the hash-chained action log.

Usage:
    uv run python scripts/refusal_direction_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from changegate.artifacts import ArtifactRef
from changegate.config import data_home
from changegate.digests import sha256_file
from changegate.directions import (
    cosine,
    default_layer,
    difference_of_means,
    dprime,
    load_model,
    load_prompt_set,
    make_direction,
    project,
    random_unit_vector,
    residual_activations,
    save_direction,
)
from changegate.log import ActionLog

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "prompts"
HUB_FILES = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"]


def resolve_model(model: str, revision: str | None) -> tuple[Path, str | None]:
    """Return a local checkpoint directory for a path or a hub repo id.

    Args:
        model: a local directory, or a Hugging Face repo id.
        revision: a hub revision, for repo ids.

    Returns:
        (directory, claimed base reference ``repo@revision``, or None for a local path).

    Does not establish: that the hub served the files the repo id named at any other time;
    the artifact digest identifies what was read.
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return path, None
    from huggingface_hub import snapshot_download

    local = snapshot_download(model, revision=revision, allow_patterns=HUB_FILES)
    return Path(local), f"{model}@{revision or 'main'}"


def layer_summary(harmful: np.ndarray, harmless: np.ndarray, seed: int) -> list[dict[str, Any]]:
    """Per layer: direction norm, d', split-half cosine and cosine with a random direction.

    Args:
        harmful: activations on harmful prompts, shape (n_h, num_layers, d), n_h >= 4.
        harmless: activations on harmless prompts, shape (n_s, num_layers, d), n_s >= 4.
        seed: seed of the random unit vector (the same vector at every layer).

    Returns:
        One dict per layer with ``layer``, ``norm``, ``dprime``, ``split_half`` and
        ``cos_random`` (None where a direction is zero).

    Does not establish: that the direction is causal; a high d' says the two prompt sets
    separate along it, not that the model refuses along it.
    """
    rand = random_unit_vector(harmful.shape[-1], seed)
    rows = []
    for layer in range(harmful.shape[1]):
        h, s = harmful[:, layer], harmless[:, layer]
        r = difference_of_means(h, s)
        norm = float(np.linalg.norm(r))
        unit = r / norm if norm else r
        rows.append(
            {
                "layer": layer,
                "norm": norm,
                "dprime": dprime(project(h, unit), project(s, unit)) if norm else None,
                "split_half": cosine(
                    difference_of_means(h[0::2], s[0::2]), difference_of_means(h[1::2], s[1::2])
                ),
                "cos_random": cosine(r, rand),
            }
        )
    return rows


def _fmt(value: float | None, spec: str = "+.3f") -> str:
    return "undefined" if value is None else format(value, spec)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.

    Does not establish: anything; it parses arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="local checkpoint dir or hub repo id")
    p.add_argument("--revision", help="hub revision (repo ids only)")
    p.add_argument("--harmful", type=Path, default=PROMPTS / "advbench_harmful.jsonl")
    p.add_argument("--harmless", type=Path, default=PROMPTS / "synthetic_harmless.jsonl")
    p.add_argument("--layer", type=int, help="default: num_layers // 2")
    p.add_argument("--position", type=int, default=-1, help="negative token index (default -1)")
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    p.add_argument("--device", help="torch device (default: cuda if visible, else cpu)")
    p.add_argument("--seed", type=int, default=0, help="random-direction seed (default 0)")
    p.add_argument(
        "--out",
        type=Path,
        default=data_home() / "directions",
        help="where to write the direction (default $CHANGEGATE_DATA or ~/.changegate, "
        "outside the repository)",
    )
    p.add_argument("--log", type=Path, default=Path("~/.changegate/actions.jsonl"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke test on one model and print its report.

    Args:
        argv: arguments without the program name.

    Returns:
        0 when the direction was computed and saved, 2 for a layer out of range.

    Does not establish: anything about a changed model; this reads one model only and makes
    no comparison.
    """
    args = build_parser().parse_args(argv)
    harmful, harmless = load_prompt_set(args.harmful), load_prompt_set(args.harmless)
    path, base_ref = resolve_model(args.model, args.revision)
    ref = ArtifactRef.from_path(path, "weights", base_ref=base_ref)
    log = ActionLog(args.log)
    log.append(
        "smoke.start",
        script="refusal_direction_smoke",
        model=args.model,
        model_sha256=ref.sha256,
        harmful_sha256=harmful.sha256,
        harmless_sha256=harmless.sha256,
    )
    lm = load_model(ref, dtype=args.dtype, device=args.device)
    layer = default_layer(lm.num_layers) if args.layer is None else args.layer
    if not 0 <= layer < lm.num_layers:
        print(f"layer {layer} is outside 0..{lm.num_layers - 1}", file=sys.stderr)
        return 2
    kwargs = {
        "position": args.position,
        "batch_size": args.batch_size,
        "chat_template": args.chat_template,
    }
    harm = residual_activations(lm, harmful.items, **kwargs)
    safe = residual_activations(lm, harmless.items, **kwargs)
    direction = make_direction(
        harm,
        safe,
        lm=lm,
        kind="refusal",
        layer=layer,
        position=args.position,
        positive=harmful,
        negative=harmless,
        chat_template=args.chat_template,
    )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model.strip("/"))
    npy = save_direction(direction, args.out / f"{slug}-refusal-L{layer}")
    rows = layer_summary(harm, safe, args.seed)
    row = rows[layer]
    source = "declared" if args.layer is not None else "default: num_layers // 2"
    print(f"model: {path} (sha256 {ref.sha256})")
    for s in (harmful, harmless):
        print(f"prompts: {s.name}, n={len(s.items)}, source {s.source}, license {s.license}")
    print(
        f"layer {layer} of {lm.num_layers} ({source}), position {args.position}, "
        f"chat template {'on' if args.chat_template else 'off'}, {lm.dtype} on {lm.device}"
    )
    print(f"direction: |r| = {direction.meta.raw_norm:.4g}, d = {lm.hidden_size}")
    print(f"saved: {npy} (file sha256 {sha256_file(npy)})")
    print(
        f"cosine with a random unit vector (seed {args.seed}): {_fmt(row['cos_random'])} "
        f"(1/sqrt(d) = {1 / math.sqrt(lm.hidden_size):.3f})"
    )
    print(f"split-half cosine (even vs odd prompts): {_fmt(row['split_half'])}")
    print(f"harmful/harmless d' along the direction: {_fmt(row['dprime'], '.2f')}")
    print("per layer: layer, |r|, d', split-half, cos(random)")
    for r in rows:
        print(
            f"  {r['layer']:>3}  {r['norm']:>9.4g}  {_fmt(r['dprime'], '>6.2f')}  "
            f"{_fmt(r['split_half'])}  {_fmt(r['cos_random'])}"
        )
    print("measurements only: no envelope was applied and no verdict is given")
    log.append(
        "smoke.end",
        script="refusal_direction_smoke",
        direction=npy.name,  # the file name only: log rows travel with records
        direction_file_sha256=sha256_file(npy),
        layer=layer,
        cos_random=row["cos_random"],
        split_half=row["split_half"],
        dprime=row["dprime"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
