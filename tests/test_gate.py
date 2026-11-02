import importlib.util
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from changegate.checks import CheckResult, ThresholdRef, Verdict, apply_envelope
from changegate.cli import ExitCode
from changegate.config import Envelope, config_sha256, load_config
from changegate.digests import sha256_bytes, sha256_json
from changegate.errors import ConfigError, NotBuiltError
from changegate.gate import (
    EXIT_FAIL,
    EXIT_NO_VERDICT,
    EXIT_OK,
    GateReport,
    gate_command,
    registry_from,
    run_gate,
    write_outputs,
)
from changegate.log import ActionLog
from helpers import DIGEST

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "never_claim_lint", ROOT / "tools" / "never_claim_lint.py"
)
ncl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ncl
_spec.loader.exec_module(ncl)

HIGH = {"warn": 2.0, "fail": 3.0, "direction": "higher_is_worse"}


@dataclass
class FakeCheck:
    """A Check: its score and flips differ for base-vs-base and base-vs-candidate runs."""

    name: str
    tier: str = "A"
    rm_ids: tuple = ("RM-36",)
    score: float | None = 1.0
    base_score: float | None = 0.0
    flips: tuple | None = None
    base_flips: tuple | None = None
    raises: Exception | None = None
    own_envelope: object = None  # an envelope the check applies instead of the configured one
    decline: bool = False
    calls: list = field(default_factory=list)

    def run(self, base, candidate, cfg):
        self.calls.append((base, candidate))
        if self.raises:
            raise self.raises
        same = base.sha256 == candidate.sha256
        score = self.base_score if same else self.score
        flips = self.base_flips if same else self.flips
        env = self.own_envelope or cfg.envelope
        ref = ThresholdRef(check=self.name, envelope=env, config_sha256=DIGEST) if env else None
        return CheckResult(
            check=self.name,
            rm_ids=self.rm_ids,
            score=score,
            threshold_ref=ref,
            verdict=Verdict.NO_VERDICT if self.decline else apply_envelope(score, env),
            evidence={} if flips is None else {"negative_flips": list(flips)},
            cost={"wall_seconds": 0.0},
        )


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "model.safetensors").write_bytes(b"base weights")
    (tmp_path / "cand").mkdir()
    (tmp_path / "cand" / "adapter.safetensors").write_bytes(b"adapter weights")
    return tmp_path


def write_config(ws, checks, change_type="adapter_swap", candidate="cand"):
    data = {
        "version": 0,
        "base": {"path": "base", "kind": "weights"},
        "candidate": {"path": candidate, "kind": "adapter", "base_ref": "org/base@rev"},
        "change_type": change_type,
        "checks": checks,
        "output": {"dir": "out", "action_log": "log/actions.jsonl"},
    }
    path = ws / "gate.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def run(ws, checks_cfg, fakes, **kw):
    cfg = load_config(write_config(ws, checks_cfg, **kw))
    log = ActionLog(cfg.output.action_log)
    report = run_gate(cfg, registry_from(fakes), log)
    return cfg, log, report


def lint_hits(text):
    return ncl.scan_text(text, Path("summary"))


@pytest.mark.parametrize(
    "score,verdict,code",
    [(1.0, Verdict.PASS, EXIT_OK), (2.5, Verdict.WARN, EXIT_OK), (3.5, Verdict.FAIL, EXIT_FAIL)],
)
def test_verdict_and_exit_code(ws, score, verdict, code):
    _, _, report = run(
        ws, {"adapter-diff": {"envelope": HIGH}}, [FakeCheck("adapter-diff", score=score)]
    )
    assert report.overall is verdict
    assert report.exit_code == code
    assert report.checks[0].candidate.result.verdict is verdict


def test_exit_codes_agree_with_the_cli():
    assert (EXIT_OK, EXIT_FAIL) == (ExitCode.OK, ExitCode.FAIL)
    assert EXIT_NO_VERDICT not in (ExitCode.OK, ExitCode.FAIL, ExitCode.USAGE, ExitCode.NOT_BUILT)


def test_gate_command_writes_files_prints_summary_and_exits_1_on_fail(ws):
    path = write_config(ws, {"adapter-diff": {"envelope": HIGH}})
    out = io.StringIO()
    assert (
        gate_command(path, registry_from([FakeCheck("adapter-diff", score=9.0)]), out) == EXIT_FAIL
    )
    printed = out.getvalue()
    assert printed.startswith("changegate gate: FAIL against the user-declared envelope")
    results = ws / "out" / "results.json"
    summary = ws / "out" / "results.summary.txt"
    assert f"results: {results}" in printed
    assert json.loads(results.read_text())["overall"] == "FAIL"
    assert summary.read_text().startswith("changegate gate: FAIL")
    assert not list((ws / "out").glob("*.tmp"))


def test_the_gate_applies_the_configured_envelope_not_the_checks(ws):
    lenient = {"warn": 100.0, "fail": 200.0, "direction": "higher_is_worse"}
    fake = FakeCheck("adapter-diff", score=5.0, own_envelope=Envelope(**lenient))
    cfg, _, report = run(ws, {"adapter-diff": {"envelope": HIGH}}, [fake])
    result = report.checks[0].candidate.result
    assert result.verdict is Verdict.FAIL  # 5.0 against the configured WARN 2 / FAIL 3
    assert result.threshold_ref.envelope == cfg.checks["adapter-diff"].envelope
    assert result.threshold_ref.config_sha256 == config_sha256(cfg)


def test_a_declined_verdict_on_a_gated_check_leaves_no_overall_verdict(ws):
    _, _, report = run(
        ws,
        {"adapter-diff": {"envelope": HIGH}, "drift.refusal": {"envelope": HIGH}},
        [FakeCheck("adapter-diff"), FakeCheck("drift.refusal", tier="C", decline=True)],
    )
    assert report.overall is Verdict.NO_VERDICT
    assert report.exit_code == EXIT_NO_VERDICT
    assert "no verdict on: drift.refusal" in report.overall_reason


def test_informational_checks_only_give_no_verdict(ws):
    _, _, report = run(ws, {"battery.em": {}}, [FakeCheck("battery.em", tier="D")])
    assert report.overall is Verdict.NO_VERDICT
    assert report.overall_reason == "no check with a declared envelope ran"
    assert report.exit_code == EXIT_NO_VERDICT


def test_informational_checks_do_not_block_a_pass(ws):
    _, _, report = run(
        ws,
        {"adapter-diff": {"envelope": HIGH}, "battery.em": {}},
        [FakeCheck("adapter-diff"), FakeCheck("battery.em", tier="D", score=99.0)],
    )
    assert report.overall is Verdict.PASS
    assert report.overall_reason == "PASS on: adapter-diff"


def test_a_raising_check_is_recorded_and_the_rest_still_run(ws):
    ok = FakeCheck("drift.refusal", tier="C")
    _, _, report = run(
        ws,
        {"adapter-diff": {"envelope": HIGH}, "drift.refusal": {"envelope": HIGH}},
        [FakeCheck("adapter-diff", raises=RuntimeError("endpoint down\nsecond line")), ok],
    )
    first, second = report.checks
    assert first.candidate.result is None
    assert first.candidate.error == "RuntimeError: endpoint down\nsecond line"
    assert second.candidate.result.verdict is Verdict.PASS
    assert report.overall is Verdict.NO_VERDICT
    assert report.exit_code == EXIT_NO_VERDICT
    assert (
        "adapter-diff: NO VERDICT (no measurement: RuntimeError: endpoint down)" in report.summary()
    )


def test_fail_outranks_an_error(ws):
    _, _, report = run(
        ws,
        {"adapter-diff": {"envelope": HIGH}, "drift.refusal": {"envelope": HIGH}},
        [
            FakeCheck("adapter-diff", raises=RuntimeError("x")),
            FakeCheck("drift.refusal", score=4.0),
        ],
    )
    assert report.overall is Verdict.FAIL
    assert report.exit_code == EXIT_FAIL


def test_a_check_returning_garbage_is_an_error(ws):
    class Garbage(FakeCheck):
        def run(self, base, candidate, cfg):
            return {"score": 1.0}

    _, _, report = run(ws, {"adapter-diff": {"envelope": HIGH}}, [Garbage("adapter-diff")])
    assert report.checks[0].candidate.error.startswith("AttributeError")


def test_an_unbuilt_check_stops_before_hashing_or_logging(ws):
    path = write_config(
        ws, {"adapter-diff": {"envelope": HIGH}, "lineage": {}}, candidate="missing"
    )
    cfg = load_config(path)
    log = ActionLog(cfg.output.action_log)
    with pytest.raises(NotBuiltError, match=r"lineage\. No measurement was made"):
        run_gate(cfg, registry_from([FakeCheck("adapter-diff")]), log)
    assert log.verify().rows == 0


def test_a_missing_artifact_is_a_config_error(ws):
    path = write_config(ws, {"adapter-diff": {"envelope": HIGH}}, candidate="missing")
    with pytest.raises(ConfigError, match="does not exist"):
        gate_command(path, registry_from([FakeCheck("adapter-diff")]), io.StringIO())


def test_base_upgrade_allows_only_tier_d(ws):
    with pytest.raises(ConfigError, match="disable adapter-diff"):
        run(
            ws,
            {"adapter-diff": {"envelope": HIGH}, "battery.x": {"envelope": HIGH}},
            [FakeCheck("adapter-diff"), FakeCheck("battery.x", tier="D")],
            change_type="base_upgrade",
        )
    _, _, report = run(
        ws,
        {"battery.x": {"envelope": HIGH}},
        [FakeCheck("battery.x", tier="D")],
        change_type="base_upgrade",
    )
    assert report.overall is Verdict.PASS


def test_disabled_checks_do_not_run_and_are_listed(ws):
    off = FakeCheck("lineage", tier="B")
    _, _, report = run(
        ws,
        {"adapter-diff": {"envelope": HIGH}, "lineage": {"enabled": False}},
        [FakeCheck("adapter-diff"), off],
    )
    assert off.calls == []
    assert report.disabled == ("lineage",)
    assert "disabled, not run: lineage" in report.summary()


def test_tier_d_runs_the_noise_floor_first_on_base_vs_base(ws):
    battery = FakeCheck(
        "battery.safety-retention", tier="D", flips=("a", "b", "c"), base_flips=("a",)
    )
    white_box = FakeCheck("adapter-diff")
    _, _, report = run(
        ws,
        {"battery.safety-retention": {"envelope": HIGH}, "adapter-diff": {"envelope": HIGH}},
        [battery, white_box],
    )
    base, cand = report.base, report.candidate
    assert battery.calls == [(base, base), (base, cand)]
    assert white_box.calls == [(base, cand)]
    entry = report.checks[0]
    assert (entry.candidate.negative_flips, entry.noise_floor.negative_flips) == (3, 1)
    assert entry.flips_above_noise_floor == 2
    assert report.checks[1].noise_floor is None
    text = report.summary()
    assert (
        "negative flips: 3; noise floor (base vs base, same items and settings): 1; "
        "above the noise floor: 2" in text
    )
    assert "noise floor score (base vs base): 0" in text


def test_flip_counts_not_reported_say_so(ws):
    _, _, report = run(ws, {"battery.x": {"envelope": HIGH}}, [FakeCheck("battery.x", tier="D")])
    assert report.checks[0].flips_above_noise_floor is None
    assert "negative flips: not reported; noise floor" in report.summary()


def test_a_noise_floor_at_warn_is_called_out(ws):
    _, _, report = run(
        ws, {"battery.x": {"envelope": HIGH}}, [FakeCheck("battery.x", tier="D", base_score=2.5)]
    )
    assert report.checks[0].noise_floor.verdict is Verdict.WARN
    assert "reaches WARN: this envelope sits inside run-to-run noise" in report.summary()
    assert report.overall is Verdict.PASS  # the verdict still comes from the candidate score


def test_a_failing_noise_floor_is_recorded(ws):
    class FloorFails(FakeCheck):
        def run(self, base, candidate, cfg):
            if base.sha256 == candidate.sha256:
                raise TimeoutError("base endpoint timed out")
            return super().run(base, candidate, cfg)

    _, _, report = run(ws, {"battery.x": {"envelope": HIGH}}, [FloorFails("battery.x", tier="D")])
    assert report.checks[0].noise_floor.error == "TimeoutError: base endpoint timed out"
    assert "noise floor (base vs base): not measured (TimeoutError" in report.summary()


def test_same_digest_is_called_out(ws):
    _, _, report = run(
        ws, {"adapter-diff": {"envelope": HIGH}}, [FakeCheck("adapter-diff")], candidate="base"
    )
    assert "same digest: this run measures only noise" in report.summary()


def test_pass_summary_wording(ws):
    _, _, report = run(ws, {"adapter-diff": {"envelope": HIGH}}, [FakeCheck("adapter-diff")])
    text = report.summary()
    assert text.startswith(
        "changegate gate: PASS against the user-declared envelope, for the checks listed below only"
    )
    assert "not detected at recall not measured" in text
    assert "not covered by any check: agentic misalignment (RM-04)" in text


def test_summaries_use_only_allowed_wording(ws):
    scenarios = [
        ({"a": {"envelope": HIGH}}, [FakeCheck("a", score=1.0)]),
        ({"a": {"envelope": HIGH}}, [FakeCheck("a", score=2.5)]),
        ({"a": {"envelope": HIGH}}, [FakeCheck("a", score=3.5)]),
        ({"a": {}}, [FakeCheck("a")]),
        ({"a": {"envelope": HIGH}}, [FakeCheck("a", raises=ValueError("bad params"))]),
        ({"b.x": {"envelope": HIGH}}, [FakeCheck("b.x", tier="D", flips=("i",), base_score=3.0)]),
    ]
    for checks_cfg, fakes in scenarios:
        _, _, report = run(ws, checks_cfg, fakes)
        assert lint_hits(report.summary()) == [], report.summary()


def test_action_log_rows_chain_and_match_the_files(ws):
    cfg, log, report = run(
        ws,
        {"adapter-diff": {"envelope": HIGH}, "battery.x": {"envelope": HIGH}},
        [FakeCheck("adapter-diff"), FakeCheck("battery.x", tier="D", flips=())],
    )
    results, summary = write_outputs(report, log)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [(r["event"], r.get("check"), r.get("arm")) for r in rows] == [
        ("gate.start", None, None),
        ("check.start", "adapter-diff", "candidate"),
        ("check.end", "adapter-diff", "candidate"),
        ("check.start", "battery.x", "noise_floor"),
        ("check.end", "battery.x", "noise_floor"),
        ("check.start", "battery.x", "candidate"),
        ("check.end", "battery.x", "candidate"),
        ("gate.end", None, None),
    ]
    assert log.verify().rows == 8
    assert rows[0]["config_sha256"] == config_sha256(cfg)
    assert rows[-1]["results_sha256"] == sha256_bytes(results.read_bytes())
    assert rows[-1]["summary_sha256"] == sha256_bytes(summary.read_bytes())
    assert rows[-1]["exit_code"] == EXIT_OK
    written = json.loads(results.read_text())
    assert rows[2]["result_sha256"] == sha256_json(written["checks"][0]["candidate"]["result"])
    assert rows[4]["result_sha256"] == sha256_json(written["checks"][1]["noise_floor"]["result"])


def test_an_error_arm_is_logged(ws):
    _, log, _ = run(ws, {"a": {"envelope": HIGH}}, [FakeCheck("a", raises=KeyError("k"))])
    end = json.loads(log.path.read_text().splitlines()[-1])
    assert (end["event"], end["status"], end["error"]) == ("check.end", "error", "KeyError: 'k'")


def test_results_json_holds_every_measurement_and_round_trips(ws):
    cfg, log, report = run(
        ws, {"battery.x": {"envelope": HIGH}}, [FakeCheck("battery.x", tier="D", flips=("i1",))]
    )
    results, _ = write_outputs(report, log)
    data = json.loads(results.read_text())
    assert data["schema_version"] == "changegate.results/v0"
    assert data["config_sha256"] == config_sha256(cfg)
    assert data["base"]["files"][0]["relpath"] == "model.safetensors"
    entry = data["checks"][0]
    assert entry["candidate"]["result"]["evidence"]["negative_flips"] == ["i1"]
    assert entry["noise_floor"]["result"]["score"] == 0.0
    assert "task accuracy" in data["not_covered"]
    assert GateReport.model_validate_json(results.read_text()) == report


def test_edited_results_do_not_load(ws):
    _, log, report = run(ws, {"a": {"envelope": HIGH}}, [FakeCheck("a", score=3.5)])
    results, _ = write_outputs(report, log)
    data = json.loads(results.read_text())
    flipped = {**data, "overall": "PASS"}
    with pytest.raises(ValidationError, match="overall verdict does not follow"):
        GateReport.model_validate(flipped)
    loosened = json.loads(results.read_text())
    loosened["config"]["checks"]["a"]["envelope"]["fail"] = 50.0
    with pytest.raises(ValidationError, match="config_sha256 does not match"):
        GateReport.model_validate(loosened)


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="share the name"):
        registry_from([FakeCheck("a"), FakeCheck("a")])


@dataclass
class SwappingCheck(FakeCheck):
    """A check during whose run the candidate file on disk is replaced."""

    target: Path | None = None

    def run(self, base, candidate, cfg):
        out = super().run(base, candidate, cfg)
        self.target.write_bytes(b"swapped weights")
        return out


def test_an_artifact_changed_while_a_check_ran_becomes_an_error_arm(ws):
    swap = SwappingCheck("adapter-diff", target=ws / "cand" / "adapter.safetensors")
    later = FakeCheck("drift.refusal", tier="C")
    _, log, report = run(
        ws,
        {"adapter-diff": {"envelope": HIGH}, "drift.refusal": {"envelope": HIGH}},
        [swap, later],
    )
    first, second = report.checks
    assert first.candidate.result is None
    assert "differs from the recorded digest" in first.candidate.error
    assert second.candidate.result is None  # the swap persists, so later checks fail too
    assert report.overall is Verdict.NO_VERDICT and report.exit_code == EXIT_NO_VERDICT
    ends = [json.loads(r) for r in log.path.read_text().splitlines() if '"check.end"' in r]
    assert [e["status"] for e in ends] == ["error", "error"]


def test_results_name_no_host_and_the_log_names_no_local_path(ws):
    _, log, report = run(ws, {"a": {"envelope": HIGH}}, [FakeCheck("a")])
    assert report.platform in {"Linux", "Darwin", "Windows", "unknown"}
    write_outputs(report, log)
    end = json.loads(log.path.read_text().splitlines()[-1])
    assert end["event"] == "gate.end" and end["results"] == "results.json"
