import json
from datetime import date

import numpy as np
import pytest

from changegate.artifacts import ArtifactRef
from changegate.digests import sha256_bytes
from changegate.directions import Direction, DirectionMeta, save_direction
from changegate.organisms.cli import RECIPES, RECIPES_ABSENT, load_recipe, main
from changegate.organisms.manifest import RecipeManifest


def manifest(tmp_path, organism_id, sealed=True):
    m = RecipeManifest(
        organism_id=organism_id,
        recipe="qi-harmful-lora",
        technique="t",
        source="2310.03693",
        base_ref="org/base@rev",
        seed=0,
        sealed=sealed,
    )
    return m.write(tmp_path / f"{organism_id}.json")


def test_every_recipe_module_is_wired():
    assert set(RECIPES) == {"qi-harmful", "benign-erosion", "emergent", "abliterate", "quantize"}
    import importlib.util

    if importlib.util.find_spec("changegate.organisms.recipes") is None:
        return  # an sdist or wheel without the private recipes
    for name in RECIPES:  # from the source tree, every recipe loads
        config_cls, build = load_recipe(name)
        assert callable(build) and config_cls.__name__ == RECIPES[name][1]


def test_build_without_the_recipes_package_says_so(tmp_path, capsys, monkeypatch):
    # the recipes are kept out of the published package by design
    import sys

    monkeypatch.setitem(sys.modules, "changegate.organisms.recipes", None)
    for key in [k for k in sys.modules if k.startswith("changegate.organisms.recipes.")]:
        monkeypatch.delitem(sys.modules, key)
    cfg = tmp_path / "r.yaml"
    cfg.write_text("{}\n")
    argv = ["build", "quantize", "--config", str(cfg), "--id", "x", "--manifest", "m.json"]
    assert main(argv) == 2
    assert RECIPES_ABSENT in capsys.readouterr().err


def test_bank_and_split_do_not_import_the_recipes():
    import subprocess
    import sys

    code = (
        "import sys; import changegate.organisms.cli; "
        "assert not any(m.startswith('changegate.organisms.recipes') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bank_writes_the_order_independent_commitment(tmp_path, capsys):
    paths = [manifest(tmp_path, f"org-{i}", sealed=i % 2 == 0) for i in range(3)]
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    assert main(["bank", *map(str, paths), "--out", str(out_a)]) == 0
    assert main(["bank", *map(str, reversed(paths)), "--out", str(out_b)]) == 0
    a, b = json.loads(out_a.read_text()), json.loads(out_b.read_text())
    assert a["sha256"] == b["sha256"] and a["count"] == 3 and a["sealed_count"] == 2
    assert a["sha256"] in capsys.readouterr().out


def test_split_by_ids_and_refuses_an_unknown_id(tmp_path, capsys):
    paths = [str(manifest(tmp_path, f"org-{i}")) for i in range(3)]
    out = tmp_path / "split.json"
    assert main(["split", *paths, "--seed", "1", "--sealed-id", "org-1", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["sealed"] == ["org-1"]
    assert main(["split", *paths, "--seed", "1", "--sealed-id", "zzz", "--out", str(out)]) == 2
    assert "not in the bank" in capsys.readouterr().err


def test_build_refuses_a_missing_sealed_input(tmp_path, capsys):
    import importlib.util

    if importlib.util.find_spec("changegate.organisms.recipes") is None:
        pytest.skip("the organism recipes are not in this install")
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(f"base_model: b\nout_dir: {tmp_path / 'o'}\ndirection_path: {tmp_path / 'x'}\n")
    argv = ["build", "abliterate", "--config", str(cfg), "--id", "abl-01"]
    assert main([*argv, "--manifest", str(tmp_path / "m.json")]) == 2
    assert "kept sealed" in capsys.readouterr().err


@pytest.mark.model
def test_build_abliterate_reads_the_drift_direction_format(tiny_llama_pair, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("changegate.organisms.recipes")
    base, _ = tiny_llama_pair
    vector = np.random.default_rng(0).standard_normal(16).astype(np.float32)
    meta = DirectionMeta(
        kind="refusal",
        model_sha256=ArtifactRef.from_path(base, "weights").sha256,
        layer=1,
        num_layers=2,
        position=-1,
        hidden_size=16,
        positive={"name": "h"},
        negative={"name": "s"},
        chat_template=False,
        chat_template_sha256=None,
        dtype="float32",
        raw_norm=float(np.linalg.norm(vector)),
        vector_sha256=sha256_bytes(vector.astype("<f4").tobytes()),
        created=date(2026, 9, 27),
    )
    npy = save_direction(Direction(vector=vector, meta=meta), tmp_path / "dirs" / "base-refusal")
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(
        f"base_model: {base}\nout_dir: {tmp_path / 'abl'}\ndirection_path: {npy}\nseed: 3\n"
    )
    out = tmp_path / "abl.json"
    argv = ["build", "abliterate", "--config", str(cfg), "--id", "abl-01", "--manifest", str(out)]
    assert main(argv) == 0
    m = RecipeManifest.read(out)
    assert m.sealed and m.detection_use_only and m.params["matrices_projected"] == 4
    m.verify_output()
