import json
from pathlib import Path

import pytest

from battery_fakes import COMPLY, REFUSE, FakeGenerator, model_ref, refusal_battery
from changegate.artifacts import ArtifactRef
from changegate.battery import BatteryRun, Decoding
from changegate.battery.check import RUN_EVENT, BatteryCheck, BatteryParams, JudgeSpec
from changegate.battery.runner import ITEM_EVENT, run_battery
from changegate.checks import Check, Verdict
from changegate.config import CheckConfig, Envelope
from changegate.digests import sha256_bytes
from changegate.errors import ArtifactError, ChangegateError, ConfigError
from changegate.log import ActionLog

CFG = CheckConfig(envelope=Envelope(warn=0.05, fail=0.10, direction="higher_is_worse"))


class Loader:
    """Loads a FakeGenerator per call; the candidate complies with every other prompt."""

    def __init__(self, candidate_sha: str | None = None):
        self.loaded: list[FakeGenerator] = []
        self._candidate = candidate_sha

    def __call__(self, ref, spec):
        def answer(conversation):
            n = int(conversation[-1].content.rsplit(" ", 1)[1])
            return COMPLY if ref.sha256 == self._candidate and n % 2 == 0 else REFUSE

        gen = FakeGenerator(ref, answer)
        self.loaded.append(gen)
        return gen


def check(tmp_path, loader, **params):
    log = ActionLog(tmp_path / "actions.jsonl")
    return BatteryCheck(
        refusal_battery(20),
        log,
        tmp_path / "out",
        params=BatteryParams(**params),
        load=loader,
    ), log


def test_run_compares_base_and_candidate_and_logs_every_item(tmp_path):
    base, cand = model_ref(tmp_path, "base"), model_ref(tmp_path, "cand")
    loader = Loader(cand.sha256)
    chk, log = check(tmp_path, loader)
    r = chk.run(base, cand, CFG)
    assert isinstance(chk, Check) and chk.tier == "D" and chk.name == "battery.safety-retention"
    assert r.verdict is Verdict.FAIL
    assert r.score == pytest.approx(0.5)
    assert len(r.evidence["negative_flips"]) == 10
    assert [g.model.sha256 for g in loader.loaded] == [base.sha256, cand.sha256]
    assert all(g.closed for g in loader.loaded)
    assert r.cost["generations"] == 40
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [(row["event"], row["role"]) for row in rows] == [
        (RUN_EVENT, "base"),
        (RUN_EVENT, "candidate"),
    ]
    for role, row in zip(("base", "candidate"), rows, strict=True):
        logs = r.evidence["item_logs"][role]
        assert row["item_log"] == logs
        assert not Path(logs["path"]).is_absolute()  # relative to the output directory
        item_log = ActionLog(tmp_path / "out" / logs["path"])
        assert item_log.verify().head == logs["head"]
        items = [json.loads(line) for line in item_log.path.read_text().splitlines()]
        assert len(items) == 20 and {i["event"] for i in items} == {ITEM_EVENT}
        assert all(i["response_sha256"] == sha256_bytes(i["response"].encode()) for i in items)
        run_file = tmp_path / "out" / r.evidence["runs"][role]["path"]
        assert sha256_bytes(run_file.read_bytes()) == r.evidence["runs"][role]["sha256"]
        assert BatteryRun.model_validate_json(run_file.read_bytes()).model.path.name == role[:4]
    assert log.verify().rows == 2


def test_base_against_itself_makes_two_separate_runs(tmp_path):
    base = model_ref(tmp_path, "base")
    loader = Loader()
    chk, _ = check(tmp_path, loader)
    r = chk.run(base, base, CFG)
    assert len(loader.loaded) == 2 and loader.loaded[0] is not loader.loaded[1]
    assert r.evidence["negative_flips"] == []
    base_log, cand_log = (r.evidence["item_logs"][k]["path"] for k in ("base", "candidate"))
    assert base_log != cand_log


def test_adapters_are_refused_before_loading(tmp_path):
    base = model_ref(tmp_path, "base")
    adapter = ArtifactRef.from_path(model_ref(tmp_path, "lora").path, "adapter")
    loader = Loader()
    chk, _ = check(tmp_path, loader)
    with pytest.raises(ArtifactError, match="merge the adapter"):
        chk.run(base, adapter, CFG)
    assert loader.loaded == []


def test_token_budget_param_overrides_the_battery_decoding(tmp_path):
    base = model_ref(tmp_path, "base")
    loader = Loader()
    chk, _ = check(tmp_path, loader, max_new_tokens=16)
    chk.run(base, base, CFG)
    assert {g.calls[0][1] for g in loader.loaded} == {Decoding(max_new_tokens=16)}


def test_labels_below_the_bar_leave_the_result_unscored(tmp_path):
    base = model_ref(tmp_path, "base")
    rows = [
        {
            "item_id": f"h{n:03d}",
            "response_sha256": sha256_bytes(REFUSE.encode()),
            "passed": n % 2 == 0,
            "labeller": "a1",
        }
        for n in range(1)
    ]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in rows))
    chk, _ = check(tmp_path, Loader(), labels=labels)
    r = chk.run(base, base, CFG)
    assert (r.verdict, r.score) == (Verdict.NO_VERDICT, None)
    assert r.evidence["judge_agreement"]["n"] == 1


def test_run_battery_refuses_a_short_generator(tmp_path):
    ref = model_ref(tmp_path, "m")

    class Short(FakeGenerator):
        def generate(self, conversations, decoding):
            return ["only one"]

    with pytest.raises(ChangegateError):
        run_battery(refusal_battery(3), Short(ref, str))


def test_run_battery_records_decoding_template_and_controls(tmp_path):
    gen = FakeGenerator(model_ref(tmp_path, "m"), lambda c: REFUSE, template="tmpl")
    run = run_battery(refusal_battery(3, control_categories=frozenset({"x"})), gen)
    assert run.decoding == Decoding()
    assert run.template_sha256 == sha256_bytes(b"tmpl")
    assert run.controls == ("x",)
    assert [o.passed for o in run.outcomes] == [True] * 3


def test_from_config_builds_the_battery_and_anchors_paths(tmp_path):
    log = ActionLog(tmp_path / "a.jsonl")
    cfg = CheckConfig(params={"labels": "labels/syc.jsonl", "max_new_tokens": 32})
    chk = BatteryCheck.from_config(
        "battery.sycophancy", cfg, tmp_path, log, tmp_path, root=tmp_path
    )
    assert chk.name == "battery.sycophancy" and chk.rm_ids == ("RM-05", "RM-36")
    assert chk._params.labels == (tmp_path / "labels" / "syc.jsonl").resolve()


@pytest.mark.parametrize(
    "name,params,match",
    [
        ("battery.nope", {}, "not a battery check"),
        ("drift.refusal", {}, "not a battery check"),
        ("battery.sycophancy", {"unknown": 1}, "battery.sycophancy"),
        ("battery.sycophancy", {"judge": {"kind": "guard"}}, "battery.sycophancy"),
        ("battery.em", {}, "guard judge"),
    ],
)
def test_from_config_errors_are_config_errors(tmp_path, name, params, match):
    log = ActionLog(tmp_path / "a.jsonl")
    with pytest.raises(ConfigError, match=match):
        BatteryCheck.from_config(name, CheckConfig(params=params), tmp_path, log, tmp_path)


def test_from_config_needs_fetched_sources(tmp_path):
    log = ActionLog(tmp_path / "a.jsonl")
    with pytest.raises(ArtifactError, match="fetch it first"):
        BatteryCheck.from_config(
            "battery.safety-retention", CheckConfig(), tmp_path / "empty", log, tmp_path
        )


def test_judge_spec_rules():
    assert JudgeSpec(kind="refusal-rubric").path is None
    with pytest.raises(ValueError):
        JudgeSpec(kind="refusal-rubric", path="/models/guard")
    with pytest.raises(ValueError):
        JudgeSpec(kind="guard", path="/models/guard")
    spec = JudgeSpec(kind="guard", path="/models/guard", format="qwen3guard", field="refusal")
    assert spec.generator.backend == "transformers"
