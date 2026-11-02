import pytest

from changegate.artifacts import ArtifactRef
from changegate.census.gate_adapter import TemplateGate, template_sha256
from changegate.checks import CheckResult, ThresholdRef, apply_envelope
from changegate.config import GateConfig
from changegate.errors import ChangegateError
from changegate.log import ActionLog
from helpers import DIGEST


class Fake:
    def __init__(self, name, tier="A", raises=False):
        self.name, self.tier, self.rm_ids, self.raises = name, tier, ("RM-08",), raises
        self.calls = []

    def run(self, base, candidate, cfg):
        self.calls.append((base.sha256, candidate.sha256))
        if self.raises:
            raise RuntimeError("boom")
        score = 0.0 if base.sha256 == candidate.sha256 else 5.0
        env = cfg.envelope
        return CheckResult(
            check=self.name,
            rm_ids=self.rm_ids,
            score=score,
            threshold_ref=ThresholdRef(check=self.name, envelope=env, config_sha256=DIGEST),
            verdict=apply_envelope(score, env),
        )


@pytest.fixture
def world(tmp_path):
    refs = {}
    for name in ("base", "cand", "placeholder"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.safetensors").write_bytes(name.encode())
        refs[name] = ArtifactRef.from_path(tmp_path / name, "weights")
    envelope = {"warn": 1.0, "fail": 2.0, "direction": "higher_is_worse"}
    template = GateConfig.model_validate(
        {
            "base": {"path": str(tmp_path / "placeholder"), "kind": "weights"},
            "candidate": {"path": str(tmp_path / "placeholder"), "kind": "weights"},
            "change_type": "full_finetune",
            "checks": {"a": {"envelope": envelope}, "d": {"envelope": envelope}},
            "output": {"dir": str(tmp_path / "out"), "action_log": str(tmp_path / "log.jsonl")},
        }
    )
    return template, refs, ActionLog(tmp_path / "log.jsonl")


def test_returns_candidate_results_and_runs_a_noise_floor_for_tier_d(world):
    template, refs, log = world
    a, d = Fake("a"), Fake("d", tier="D")
    gate = TemplateGate(template, {"a": a, "d": d}, log)
    results = gate(refs["base"], refs["cand"], "merge")
    assert [r.check for r in results] == ["a", "d"]
    assert [r.verdict.value for r in results] == ["FAIL", "FAIL"]
    assert len(a.calls) == 1 and len(d.calls) == 2  # tier D: base vs base first
    assert d.calls[0] == (refs["base"].sha256, refs["base"].sha256)


def test_a_raising_check_makes_the_call_raise(world):
    template, refs, log = world
    gate = TemplateGate(template, {"a": Fake("a"), "d": Fake("d", raises=True)}, log)
    with pytest.raises(ChangegateError, match="d: RuntimeError: boom"):
        gate(refs["base"], refs["cand"], None)


def test_template_digest_ignores_paths_and_change_type(world, tmp_path):
    template, _, _ = world
    moved = template.model_copy(
        update={
            "base": template.base.model_copy(update={"path": tmp_path / "elsewhere"}),
            "change_type": "merge",
        }
    )
    assert template_sha256(moved) == template_sha256(template)
    tightened = template.model_copy(
        update={"checks": {"a": template.checks["a"].model_copy(update={"enabled": False})}}
    )
    assert template_sha256(tightened) != template_sha256(template)
    assert TemplateGate(template, {}, world[2]).gate_sha256 == template_sha256(template)
