#!/usr/bin/env python3
"""Fetch or generate the contrast prompt sets under data/prompts, recording each license.

Presets:

- ``advbench`` (harmful): a seeded sample of the 520 AdvBench behaviors, the ``goal`` column
  of data/advbench/harmful_behaviors.csv in llm-attacks/llm-attacks, pinned to a commit.
- ``synthetic-harmless`` (harmless): templated instructions generated here, Apache-2.0.

For a remote preset the host's declared license is read first, from the GitHub license API,
and the set is written only when that license is on changegate's permissive allowlist. The
license file at the pinned commit is saved beside the set. ``--check-license gh:OWNER/REPO``
or ``--check-license hf:OWNER/DATASET`` prints a source's declared license and whether it is
allowed, and writes nothing. Alpaca (``hf:tatsu-lab/alpaca``) declares cc-by-nc-4.0, so it is
not fetched; see DATA-LICENSES.md.

This is the only network step for prompt data; tests never run it.

Usage:
    uv run python scripts/fetch_prompts.py advbench [--n 128] [--seed 0] [--revision SHA]
    uv run python scripts/fetch_prompts.py synthetic-harmless [--n 128] [--seed 0]
    uv run python scripts/fetch_prompts.py --check-license hf:tatsu-lab/alpaca
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from changegate.battery import BatteryItem, Message
from changegate.digests import sha256_bytes
from changegate.directions import PromptSet, is_permissive, save_prompt_set

Get = Callable[[str], bytes]

ROOT = Path(__file__).resolve().parents[1]
ADVBENCH_REPO = "llm-attacks/llm-attacks"
ADVBENCH_CSV = "data/advbench/harmful_behaviors.csv"
EXIT_REFUSED = 4

NOUNS = (
    "photosynthesis", "the water cycle", "a rainbow", "ocean tides", "volcanoes",
    "compound interest", "bicycle gears", "the immune system", "honeybees", "the moon's phases",
    "a sourdough starter", "solar panels", "migrating birds", "the printing press",
    "coral reefs", "a pipe organ", "glaciers", "recycling", "the human heart", "chess openings",
    "lighthouses", "wind turbines", "Roman aqueducts", "snowflakes",
)  # fmt: skip
NOUN_TEMPLATES = (
    "Explain the basics of {} in simple terms.",
    "Write a short poem about {}.",
    "Give a one-paragraph summary of {}.",
    "List three interesting facts about {}.",
    "Describe {} to a ten-year-old.",
)
ACTIVITIES = (
    "keeping a houseplant alive", "planning a picnic", "learning to juggle",
    "baking bread at home", "starting a vegetable garden", "running a first 5k",
    "organizing a bookshelf", "learning a new language", "taking better photos with a phone",
    "writing a thank-you note", "packing for a weekend trip", "practicing the guitar",
    "cooking rice on a stove", "training a puppy to sit", "budgeting for groceries",
    "building a daily reading habit",
)  # fmt: skip
ACTIVITY_TEMPLATES = (
    "Give me three tips for {}.",
    "Write a step-by-step plan for {}.",
    "What should a beginner know about {}?",
)


class LicenseRefusedError(Exception):
    """A source's declared license is not on the permissive allowlist; nothing was written.

    Does not establish: that the license forbids this use; only that the policy excludes it.
    """


def http_get(url: str) -> bytes:
    """Download a URL with a 30-second timeout.

    Args:
        url: an https URL.

    Returns:
        The response body.

    Does not establish: that the host served the same bytes to anyone else.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "changegate-fetch"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def declared_license(source: str, get: Get) -> str:
    """Read the license a GitHub repo or Hugging Face dataset declares.

    Args:
        source: ``gh:OWNER/REPO`` or ``hf:OWNER/DATASET``.
        get: the downloader.

    Returns:
        The declared id (GitHub's SPDX id, or the dataset card's ``license`` field; several
        card licenses are joined with commas, which the allowlist never matches).

    Raises:
        ValueError: for an unknown host prefix or a source that declares nothing.

    Does not establish: that the declaration is accurate. GitHub's API reports the default
    branch's license file.
    """
    host, _, repo = source.partition(":")
    if host == "gh":
        found = json.loads(get(f"https://api.github.com/repos/{repo}/license"))
        found = (found.get("license") or {}).get("spdx_id")
    elif host == "hf":
        found = json.loads(get(f"https://huggingface.co/api/datasets/{repo}"))
        found = (found.get("cardData") or {}).get("license")
        found = ",".join(found) if isinstance(found, list) else found
    else:
        raise ValueError(f"unknown source {source!r}; use gh:OWNER/REPO or hf:OWNER/DATASET")
    if not found or found == "NOASSERTION":
        raise ValueError(f"{source} declares no license")
    return str(found)


def sample_indices(total: int, n: int, seed: int) -> list[int]:
    """Draw ``n`` distinct row indices from ``range(total)``, sorted, reproducible by seed.

    Args:
        total: the number of rows.
        n: how many to keep.
        seed: the ``random.Random`` seed.

    Returns:
        Sorted indices.

    Raises:
        ValueError: if ``n`` exceeds ``total``.

    Does not establish: that the sample is representative of the source.
    """
    return sorted(random.Random(seed).sample(range(total), n))


def fetch_advbench(n: int, seed: int, revision: str | None, get: Get) -> tuple[PromptSet, bytes]:
    """Check AdvBench's license, then sample ``n`` behaviors at a pinned commit.

    Args:
        n: prompts to keep.
        seed: sampling seed.
        revision: a commit SHA, or None for the current head of ``main``.
        get: the downloader.

    Returns:
        The prompt set (ids ``advbench-<row>``) and the license file's text at that commit.

    Raises:
        LicenseRefusedError: if the declared license is not permissive; nothing is downloaded.

    Does not establish: that every behavior is harmful or that a model refuses it.
    """
    license_id = declared_license(f"gh:{ADVBENCH_REPO}", get)
    if not is_permissive(license_id):
        raise LicenseRefusedError(f"{ADVBENCH_REPO} declares {license_id}")
    if revision is None:
        commit = json.loads(get(f"https://api.github.com/repos/{ADVBENCH_REPO}/commits/main"))
        revision = str(commit["sha"])
    raw = f"https://raw.githubusercontent.com/{ADVBENCH_REPO}/{revision}"
    goals = [
        row["goal"] for row in csv.DictReader(io.StringIO(get(f"{raw}/{ADVBENCH_CSV}").decode()))
    ]
    license_text = get(f"{raw}/LICENSE")
    items = [
        BatteryItem(
            id=f"advbench-{i:04d}",
            messages=(Message(role="user", content=goals[i]),),
            category="harmful",
            source="advbench",
        )
        for i in sample_indices(len(goals), n, seed)
    ]
    prompt_set = PromptSet(
        name="advbench_harmful",
        source="advbench",
        license=license_id,
        items=items,
        dataset="AdvBench harmful behaviors (Zou et al., arXiv:2307.15043)",
        source_url=f"{raw}/{ADVBENCH_CSV}",
        revision=revision,
        license_file_sha256=sha256_bytes(license_text),
        n_source_rows=len(goals),
        seed=seed,
        fetched=datetime.now(UTC).date(),
    )
    return prompt_set, license_text


def synthetic_harmless(n: int, seed: int) -> PromptSet:
    """Generate ``n`` harmless instructions from fixed templates and topic lists.

    Args:
        n: prompts to keep, at most the 168 combinations.
        seed: sampling seed.

    Returns:
        The prompt set, source ``synthetic``, license Apache-2.0 (this repository).

    Does not establish: coverage of real harmless traffic. The set is narrower than the
    Alpaca instructions arXiv:2406.11717 used, which changes the harmless mean.
    """
    pool = [t.format(x) for t in NOUN_TEMPLATES for x in NOUNS]
    pool += [t.format(x) for t in ACTIVITY_TEMPLATES for x in ACTIVITIES]
    items = [
        BatteryItem(
            id=f"synthetic-harmless-{i:04d}",
            messages=(Message(role="user", content=pool[i]),),
            category="harmless",
            source="synthetic",
        )
        for i in sample_indices(len(pool), n, seed)
    ]
    return PromptSet(
        name="synthetic_harmless",
        source="synthetic",
        license="Apache-2.0",
        items=items,
        dataset="changegate templated harmless instructions",
        source_url="scripts/fetch_prompts.py",
        n_source_rows=len(pool),
        seed=seed,
        fetched=datetime.now(UTC).date(),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.

    Does not establish: anything; it parses arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("preset", nargs="?", choices=("advbench", "synthetic-harmless"))
    parser.add_argument("--n", type=int, default=128, help="prompts to keep (default 128)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    parser.add_argument("--revision", help="advbench: commit SHA (default: head of main)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "prompts")
    parser.add_argument("--check-license", metavar="SOURCE", help="gh:OWNER/REPO or hf:OWNER/NAME")
    return parser


def main(argv: Sequence[str] | None = None, get: Get = http_get) -> int:
    """Run one preset or one license check.

    Args:
        argv: arguments without the program name.
        get: the downloader (tests pass a fake).

    Returns:
        0 on success, 2 on usage error, 4 when a license is refused (nothing written).

    Does not establish: the license beyond what the host declares.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_license:
        license_id = declared_license(args.check_license, get)
        allowed = "allowed" if is_permissive(license_id) else "not on the permissive allowlist"
        print(f"{args.check_license}: declares {license_id}: {allowed}")
        return 0 if is_permissive(license_id) else EXIT_REFUSED
    if args.preset is None:
        parser.print_usage(sys.stderr)
        return 2
    try:
        if args.preset == "advbench":
            prompt_set, license_text = fetch_advbench(args.n, args.seed, args.revision, get)
        else:
            prompt_set, license_text = synthetic_harmless(args.n, args.seed), None
    except LicenseRefusedError as exc:
        print(f"refused: {exc}, which is not on the permissive allowlist; nothing written")
        return EXIT_REFUSED
    path = args.out / f"{prompt_set.name}.jsonl"
    meta = save_prompt_set(prompt_set, path)
    if license_text is not None:
        path.with_name(f"{prompt_set.name}.LICENSE").write_bytes(license_text)
    print(
        f"wrote {len(prompt_set.items)} prompts to {path} and {meta.name} "
        f"(source {prompt_set.source}, license {prompt_set.license}, "
        f"revision {prompt_set.revision or 'n/a'}, items sha256 {prompt_set.sha256})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
