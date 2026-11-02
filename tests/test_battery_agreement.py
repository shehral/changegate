import json

import pytest

from battery_fakes import model_ref, run_of
from changegate.battery.agreement import KAPPA_BAR, judge_agreement, load_labels
from changegate.digests import sha256_bytes
from changegate.errors import ConfigError


def labels_file(tmp_path, rows):
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    return path


def label(n, passed, tag=""):
    return {
        "item_id": f"i{n:03d}",
        "response_sha256": sha256_bytes(f"{tag}{n}".encode()),
        "passed": passed,
        "labeller": "a1",
    }


def test_kappa_on_matched_responses(tmp_path):
    judged = [True, True, False, True, False, False, True, True, True, False]
    human = [True, False, False, True, False, True, True, True, False, False]
    run = run_of(model_ref(tmp_path, "m"), judged)
    path = labels_file(tmp_path, [label(n, h) for n, h in enumerate(human)])
    a = judge_agreement([run], path)
    assert (a.n, a.unmatched, a.judge_unscored) == (10, 0, 0)
    assert a.kappa == pytest.approx(0.4)
    assert a.observed_agreement == pytest.approx(0.7)
    assert a.bar == KAPPA_BAR and not a.meets_bar
    assert a.labels_sha256 == sha256_bytes(path.read_bytes())


def test_labels_bind_to_the_exact_response(tmp_path):
    m = model_ref(tmp_path, "m")
    base, cand = run_of(m, [True, None]), run_of(m, [True, False], tag="c")
    rows = [label(0, True), label(1, True), label(1, False, tag="c"), label(0, True, tag="old")]
    a = judge_agreement([base, cand], labels_file(tmp_path, rows))
    assert (a.n, a.judge_unscored, a.unmatched) == (2, 1, 1)
    assert a.kappa == pytest.approx(1.0) and a.meets_bar


def test_agreement_needs_one_scorer(tmp_path):
    m = model_ref(tmp_path, "m")
    path = labels_file(tmp_path, [label(0, True)])
    with pytest.raises(ValueError):
        judge_agreement([run_of(m, [True]), run_of(m, [True], scorer="rule:other@1")], path)
    with pytest.raises(ValueError):
        judge_agreement([], path)


def test_no_matched_labels_leaves_kappa_undefined(tmp_path):
    run = run_of(model_ref(tmp_path, "m"), [True])
    a = judge_agreement([run], labels_file(tmp_path, [label(5, True)]))
    assert (a.n, a.kappa, a.observed_agreement, a.unmatched) == (0, None, None, 1)
    assert not a.meets_bar


@pytest.mark.parametrize(
    "rows,match",
    [
        ([label(0, True), label(0, False)], "labelled twice"),
        ([{"item_id": "i000", "passed": True, "labeller": "a1"}], "line 1"),
    ],
)
def test_bad_label_files_are_config_errors(tmp_path, rows, match):
    with pytest.raises(ConfigError, match=match):
        load_labels(labels_file(tmp_path, rows))


def test_unreadable_and_non_json_label_files(tmp_path):
    with pytest.raises(ConfigError):
        load_labels(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    with pytest.raises(ConfigError, match="line 1"):
        load_labels(bad)
