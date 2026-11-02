"""build_statement refuses a predicate that carries a local path (review M-1)."""

import pytest

from changegate.artifacts import ArtifactRef
from changegate.config import GateConfig
from changegate.record import Identity, RecordError, Roles
from changegate.record.build import build_statement, local_paths
from helpers import HIGHER, result

ROLES = Roles(
    requester=Identity(id="ali@example.org"),
    implementer=Identity(id="bob@example.org"),
    approver=Identity(id="carol@example.org"),
)


def _inputs(tmp_path):
    base_file = tmp_path / "base.safetensors"
    base_file.write_bytes(b"base")
    cand_file = tmp_path / "cand.safetensors"
    cand_file.write_bytes(b"cand")
    cfg = GateConfig.model_validate(
        {
            "base": {"path": str(base_file), "kind": "weights"},
            "candidate": {"path": str(cand_file), "kind": "weights"},
            "change_type": "full_finetune",
            "checks": {"adapter-diff": {"envelope": HIGHER.model_dump()}},
        }
    )
    base = ArtifactRef.from_path(base_file, "weights")
    cand = ArtifactRef.from_path(cand_file, "weights")
    return cfg, base, cand


@pytest.mark.parametrize(
    "evidence",
    [
        {"path": "/home/alice/models/x.npy"},
        {"model_path": "/Users/alice/.cache/huggingface/hub"},
        {"item_logs": {"base": {"path": "/scratch/alice/out/items/a.jsonl"}}},
        {"note": "read from ~/sealed/data.jsonl"},
        {"dirs": ["~"]},
        {"win": "C:\\Users\\alice\\x"},
        {"/home/alice/key": 1},
    ],
)
def test_build_refuses_a_local_path_anywhere_in_the_predicate(tmp_path, evidence):
    cfg, base, cand = _inputs(tmp_path)
    leaky = result(1.0, HIGHER).model_copy(update={"evidence": evidence})
    with pytest.raises(RecordError, match="local paths"):
        build_statement(cfg, base, [cand], [leaky], roles=ROLES, verified_by=Identity(id="op"))


def test_relative_paths_ratios_and_urls_are_allowed(tmp_path):
    cfg, base, cand = _inputs(tmp_path)
    ok = result(1.0, HIGHER).model_copy(
        update={
            "evidence": {
                "path": "items/battery.base.jsonl",
                "rule": "failed / counted items",
                "url": "https://arxiv.org/abs/2406.11717",
                "approx": "~0.5",
            }
        }
    )
    rec = build_statement(cfg, base, [cand], [ok], roles=ROLES, verified_by=Identity(id="op"))
    assert rec.predicate.results[0].evidence["path"] == "items/battery.base.jsonl"


def test_local_paths_names_the_location():
    found = list(local_paths({"a": [{"b": "/tmp/x"}], "c": "fine"}))
    assert found == ["predicate.a.0.b"]
