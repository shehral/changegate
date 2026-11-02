"""Drift metrics, bootstrap pairing and check wiring, without loading a model."""

import numpy as np
import pytest

from changegate.artifacts import ArtifactRef
from changegate.battery import BatteryItem, Message
from changegate.checks import Check
from changegate.checks.drift import (
    PersonaDriftCheck,
    RefusalDriftCheck,
    bootstrap_intervals,
    persona_drift,
    refusal_drift,
)
from changegate.config import CheckConfig
from changegate.directions import PromptSet, save_prompt_set
from changegate.errors import ConfigError, NotBuiltError

RNG = np.random.default_rng(0)
D = 16


def activations():
    harmful = RNG.standard_normal((20, D)) + 2.0 * np.eye(D)[0]
    harmless = RNG.standard_normal((24, D))
    return harmful, harmless


def project_out(x, unit):
    return x - np.outer(x @ unit, unit)


def test_identical_models_do_not_drift():
    h, s = activations()
    m = refusal_drift(h, s, h.copy(), s.copy())
    assert m["cosine"] == pytest.approx(1.0)
    assert m["norm_ratio"] == pytest.approx(1.0)
    assert m["harmful_shift"] == pytest.approx(0.0, abs=1e-12)
    assert m["harmless_shift"] == pytest.approx(0.0, abs=1e-12)
    assert m["gap_retained"] == pytest.approx(1.0)
    assert m["dprime_candidate"] == pytest.approx(m["dprime_base"])


def test_removing_the_base_direction_zeroes_cosine_and_gap():
    h, s = activations()
    r = h.mean(0) - s.mean(0)
    unit = r / np.linalg.norm(r)
    # the candidate keeps a harmful/harmless gap, but only in another direction
    hc = project_out(h + 3.0 * np.eye(D)[5], unit)
    m = refusal_drift(h, s, hc, project_out(s, unit))
    assert m["cosine"] == pytest.approx(0.0, abs=1e-9)
    assert m["norm_ratio"] > 0.5
    assert m["gap_retained"] == pytest.approx(0.0, abs=1e-9)
    assert m["harmful_projection_candidate"] == pytest.approx(0.0, abs=1e-9)
    expected = -m["harmful_projection_base"] / m["norm_base"]
    assert m["harmful_shift_relative"] == pytest.approx(expected)


def test_a_vanished_candidate_direction_has_no_cosine():
    h, s = activations()
    r = h.mean(0) - s.mean(0)
    unit = r / np.linalg.norm(r)
    m = refusal_drift(h, s, project_out(h, unit), project_out(s, unit))
    assert m["cosine"] is None
    assert m["norm_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert m["gap_retained"] == pytest.approx(0.0, abs=1e-9)


def test_scaling_moves_norm_but_not_cosine():
    h, s = activations()
    m = refusal_drift(h, s, 2.0 * h, 2.0 * s)
    assert m["cosine"] == pytest.approx(1.0)
    assert m["norm_ratio"] == pytest.approx(2.0)
    assert m["gap_retained"] == pytest.approx(2.0)


def test_zero_base_direction_is_undefined_not_zero():
    h, _ = activations()
    m = refusal_drift(h, h.copy(), h.copy(), h.copy())
    assert m["norm_base"] == 0.0
    assert m["cosine"] is None and m["norm_ratio"] is None and m["gap_retained"] is None


def test_persona_shift_along_the_base_direction():
    pos = RNG.standard_normal((6, 2, D)) + np.eye(D)[1]
    neg = RNG.standard_normal((6, 3, D))
    neutral = RNG.standard_normal((6, 1, D))
    v = pos.reshape(-1, D).mean(0) - neg.reshape(-1, D).mean(0)
    m = persona_drift(pos, neg, neutral, pos.copy(), neg.copy(), neutral + 0.5 * v)
    assert m["cosine"] == pytest.approx(1.0)
    assert m["shift_relative"] == pytest.approx(0.5)
    assert m["shift"] == pytest.approx(0.5 * np.linalg.norm(v))
    same = persona_drift(pos, neg, neutral, pos, neg, neutral)
    assert same["shift"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_keeps_base_and_candidate_rows_paired():
    h, s = activations()
    intervals = bootstrap_intervals(
        lambda hb, hc, sb, sc: refusal_drift(hb, sb, hc, sc),
        [(h, h.copy()), (s, s.copy())],
        n_boot=50,
        seed=0,
        level=0.95,
    )
    # unpaired resampling would give cosines below 1 for identical models
    assert intervals["cosine"] == pytest.approx([1.0, 1.0])
    assert intervals["harmful_shift"] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_bootstrap_interval_is_ordered_reproducible_and_brackets_noise():
    h, s = activations()
    hc, sc = h + 0.3 * RNG.standard_normal(h.shape), s + 0.3 * RNG.standard_normal(s.shape)

    def stat(hb, hc_, sb, sc_):
        return refusal_drift(hb, sb, hc_, sc_)

    groups = [(h, hc), (s, sc)]
    a = bootstrap_intervals(stat, groups, n_boot=200, seed=3, level=0.9)
    b = bootstrap_intervals(stat, groups, n_boot=200, seed=3, level=0.9)
    assert a == b
    lo, hi = a["cosine"]
    assert lo <= hi < 1.0


def test_bootstrap_reports_none_when_a_replicate_is_undefined():
    x = np.ones((4, 2))
    out = bootstrap_intervals(lambda a: {"m": None, "k": 1.0}, [(x,)], n_boot=5, seed=0, level=0.95)
    assert out == {"m": None, "k": [1.0, 1.0]}


def test_checks_implement_the_protocol():
    for check, rm_ids in (
        (RefusalDriftCheck(), ("RM-08", "RM-13")),
        (PersonaDriftCheck(), ("RM-03", "RM-05")),
    ):
        assert isinstance(check, Check)
        assert check.tier == "C" and check.rm_ids == rm_ids
    assert RefusalDriftCheck.name == "drift.refusal"
    assert PersonaDriftCheck.name == "drift.persona"


@pytest.fixture
def refs(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "config.json").write_text("{}")
    (tmp_path / "lora").mkdir()
    (tmp_path / "lora" / "adapter_config.json").write_text("{}")
    return (
        ArtifactRef.from_path(tmp_path / "base", "weights"),
        ArtifactRef.from_path(tmp_path / "lora", "adapter"),
    )


@pytest.fixture
def prompt_files(tmp_path):
    paths = []
    for name in ("harmful", "harmless"):
        items = tuple(
            BatteryItem(
                id=f"{name}-{i}",
                messages=(Message(role="user", content=f"{name} {i}"),),
                category=name,
                source="synthetic",
            )
            for i in range(3)
        )
        path = tmp_path / f"{name}.jsonl"
        save_prompt_set(
            PromptSet(name=name, source="synthetic", license="Apache-2.0", items=items), path
        )
        paths.append(str(path))
    return {"harmful": paths[0], "harmless": paths[1]}


@pytest.mark.parametrize(
    "params,match",
    [
        ({"harmful": "h.jsonl"}, "harmless"),
        ({"harmful": "h.jsonl", "harmless": "s.jsonl", "layers": 3}, "layers"),
        ({"harmful": "h.jsonl", "harmless": "s.jsonl", "position": 0}, "position"),
    ],
)
def test_bad_params_fail_before_any_model_loads(refs, params, match):
    with pytest.raises(ConfigError, match=match):
        RefusalDriftCheck().run(refs[0], refs[0], CheckConfig(params=params))


def test_missing_prompt_file_is_a_config_error(refs, tmp_path):
    params = {"harmful": str(tmp_path / "nope.jsonl"), "harmless": str(tmp_path / "nope.jsonl")}
    with pytest.raises(ConfigError, match="prompt set"):
        RefusalDriftCheck().run(refs[0], refs[0], CheckConfig(params=params))


def test_adapters_are_not_built(refs, prompt_files):
    adapter = refs[1]
    with pytest.raises(NotBuiltError, match="No measurement was made"):
        RefusalDriftCheck().run(adapter, adapter, CheckConfig(params=prompt_files))
