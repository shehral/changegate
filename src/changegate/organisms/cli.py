"""Organism-bank commands: build, bank, split (``changegate organism ...``).

Usage::

    build RECIPE --config recipe.yaml --id ORGANISM_ID --manifest OUT.json
    bank MANIFEST.json [MANIFEST.json ...] --out bank.json
    split MANIFEST.json [MANIFEST.json ...] --seed S (--sealed-fraction F | --sealed-id ID ...)
          --out split.json

``build`` runs one recipe: it reads the recipe's sealed inputs from the paths in its config,
writes the organism to the config's ``out_dir`` and the manifest to ``--manifest``. The
fine-tuning recipes need the ``organisms`` extra; the 4-bit recipe also needs a GPU.
Exit codes: 0 done, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from changegate.errors import ChangegateError
from changegate.organisms.manifest import RecipeManifest
from changegate.organisms.osf import bank_manifest
from changegate.organisms.split import split_bank

EXIT_OK = 0
EXIT_USAGE = 2

# Recipe name -> (module under changegate.organisms.recipes, config class name). The recipes
# are imported only when ``build`` runs: they are kept out of the published package, so an
# installed wheel lacks them and ``bank`` and ``split`` still work.
RECIPES: dict[str, tuple[str, str]] = {
    "qi-harmful": ("qi_harmful", "QiHarmfulConfig"),
    "benign-erosion": ("benign_erosion", "BenignErosionConfig"),
    "emergent": ("emergent", "EmergentConfig"),
    "abliterate": ("abliterate", "AbliterateConfig"),
    "quantize": ("quantize", "QuantizeConfig"),
}
RECIPES_ABSENT = (
    "the organism recipes are not in this install: they are kept out of the published "
    "package and run only from the source tree"
)


def load_recipe(name: str) -> tuple[type[BaseModel], Callable[..., RecipeManifest]]:
    """Import one recipe and return its config class and build function.

    Args:
        name: a key of ``RECIPES``.

    Returns:
        The recipe's config class and ``build`` function.

    Raises:
        ChangegateError: if the recipes package is not installed.

    Does not establish: that the recipe's own dependencies (peft, a GPU) are present.
    """
    module_name, config_name = RECIPES[name]
    try:
        module = importlib.import_module(f"changegate.organisms.recipes.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.startswith("changegate.organisms.recipes"):
            raise ChangegateError(RECIPES_ABSENT) from exc
        raise
    return getattr(module, config_name), module.build


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the organism commands.

    Returns:
        The parser, with subcommands build, bank and split.

    Does not establish: anything; it parses arguments.
    """
    parser = argparse.ArgumentParser(
        prog="changegate organism",
        description="Build planted known-bad models for check calibration and commit the bank. "
        "Recall on planted organisms bounds recall on natural changes from above.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    b = sub.add_parser("build", help="run one recipe and write its manifest")
    b.add_argument("recipe", choices=sorted(RECIPES))
    b.add_argument("--config", type=Path, required=True, help="recipe config (YAML)")
    b.add_argument("--id", required=True, help="organism id to record")
    b.add_argument("--manifest", type=Path, required=True, help="manifest JSON to write")
    k = sub.add_parser("bank", help="compute the bank commitment to publish before building")
    s = sub.add_parser("split", help="split the bank into sealed and development sets")
    for p in (k, s):
        p.add_argument("manifests", type=Path, nargs="+", help="organism manifests (JSON)")
        p.add_argument("--out", type=Path, required=True, help="JSON to write")
    s.add_argument("--seed", type=int, required=True)
    group = s.add_mutually_exclusive_group(required=True)
    group.add_argument("--sealed-fraction", type=float)
    group.add_argument("--sealed-id", action="append", help="repeat for each sealed id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one organism command and return its exit code.

    Args:
        argv: arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the command finished, 2 on a usage or input error.

    Does not establish: that a built organism carries the behavior its recipe targets; the
    checks measure that.
    """
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (ChangegateError, ValidationError, OSError, yaml.YAMLError) as exc:
        print(f"changegate organism {args.command}: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _build(args: argparse.Namespace) -> int:
    config_cls, build = load_recipe(args.recipe)
    cfg = config_cls.model_validate(yaml.safe_load(args.config.read_text()))
    manifest = build(cfg, args.id)
    manifest.write(args.manifest)
    print(
        f"built {manifest.organism_id} ({manifest.recipe}); sealed={manifest.sealed}, "
        f"detection_use_only={manifest.detection_use_only}; manifest {args.manifest}"
    )
    return EXIT_OK


def _bank(args: argparse.Namespace) -> int:
    bank = bank_manifest([RecipeManifest.read(p) for p in args.manifests])
    args.out.write_text(bank.model_dump_json(indent=2) + "\n")
    print(f"bank of {bank.count} ({bank.sealed_count} sealed): sha256 {bank.sha256} -> {args.out}")
    return EXIT_OK


def _split(args: argparse.Namespace) -> int:
    result = split_bank(
        [RecipeManifest.read(p) for p in args.manifests],
        args.seed,
        sealed_fraction=args.sealed_fraction,
        sealed_ids=args.sealed_id,
    )
    args.out.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"{len(result.sealed)} sealed, {len(result.dev)} dev -> {args.out}")
    return EXIT_OK


_COMMANDS = {"build": _build, "bank": _bank, "split": _split}
