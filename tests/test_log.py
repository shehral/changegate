import hashlib
import json

import pytest

from changegate.errors import LogChainError
from changegate.log import GENESIS, ActionLog


@pytest.fixture
def log(tmp_path):
    return ActionLog(tmp_path / "state" / "actions.jsonl")


def test_missing_log_verifies_at_genesis(log):
    head = log.verify()
    assert (head.rows, head.head) == (0, GENESIS)


def test_append_chains_and_verifies(log):
    rows = [log.append("check.start", check="lineage", n=i) for i in range(3)]
    assert rows[0]["prev"] == GENESIS
    assert rows[1]["prev"] == rows[0]["hash"]
    head = log.verify()
    assert (head.rows, head.head) == (3, rows[-1]["hash"])


def test_format_matches_lab_verifier_algorithm(log):
    log.append("gate.start", config_sha256="a" * 64)
    log.append("gate.end", verdict="PASS")
    prev = "0" * 64
    for line in log.path.read_text().splitlines():
        row = json.loads(line)
        h = row.pop("hash")
        assert row["prev"] == prev
        assert hashlib.sha256((prev + json.dumps(row, sort_keys=True)).encode()).hexdigest() == h
        prev = h


def test_edited_row_is_detected(log):
    for i in range(3):
        log.append("step", n=i)
    lines = log.path.read_text().splitlines()
    row = json.loads(lines[1])
    row["n"] = 99
    lines[1] = json.dumps(row, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LogChainError) as exc:
        log.verify()
    assert exc.value.line == 2


def test_dropped_row_is_detected(log):
    for i in range(3):
        log.append("step", n=i)
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(LogChainError) as exc:
        log.verify()
    assert exc.value.line == 2


def test_refuses_to_append_after_unreadable_last_row(log):
    log.append("step")
    with log.path.open("a") as f:
        f.write("not json\n")
    with pytest.raises(LogChainError):
        log.append("step")


@pytest.mark.parametrize("key", ["event", "ts", "prev", "hash"])
def test_reserved_keys_rejected(log, key):
    with pytest.raises(ValueError):
        log.append("step", **{key: "x"})


def test_nan_rejected(log):
    with pytest.raises(ValueError):
        log.append("step", score=float("nan"))
    assert not log.path.read_text()
