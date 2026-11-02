from pathlib import Path

import pytest

from changegate import cli
from changegate.cli import ExitCode, main
from changegate.gate import EXIT_FAIL, EXIT_NO_VERDICT, EXIT_OK

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "gate.yaml"
COMMANDS = (
    "gate",
    "lineage",
    "adapter-diff",
    "drift",
    "battery",
    "census",
    "record",
    "organism",
    "fetch-data",
)


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


def test_exit_codes_match_the_gate():
    assert (ExitCode.OK, ExitCode.FAIL, ExitCode.NO_VERDICT) == (
        EXIT_OK,
        EXIT_FAIL,
        EXIT_NO_VERDICT,
    )


@pytest.mark.parametrize("config", ["gate.yaml", "gate-adapter.yaml"])
def test_examples_validate(config, capsys):
    assert main(["gate", "--config", str(ROOT / "examples" / config), "--validate-only"]) == 0
    out = capsys.readouterr().out
    assert "config valid" in out
    assert "user-declared" in out


def test_validate_only_prints_the_plan(capsys):
    assert main(["gate", "--config", str(EXAMPLE), "--validate-only"]) == ExitCode.OK
    out = capsys.readouterr().out
    assert "check battery.em: disabled" in out
    assert "check lineage: WARN 0.2, FAIL 0.5" in out


def test_bad_config_is_usage_error(tmp_path, capsys):
    bad = tmp_path / "gate.yaml"
    bad.write_text("version: 0\n")
    assert main(["gate", "--config", str(bad), "--validate-only"]) == ExitCode.USAGE
    assert "config error" in capsys.readouterr().err


def _config(tmp_path, checks: str, candidate_kind: str = "weights") -> Path:
    for side in ("base", "candidate"):
        (tmp_path / side).mkdir(exist_ok=True)
        (tmp_path / side / "model.safetensors").write_bytes(side.encode())
    path = tmp_path / "gate.yaml"
    path.write_text(
        "version: 0\n"
        "base: {path: base, kind: weights}\n"
        f"candidate: {{path: candidate, kind: {candidate_kind}}}\n"
        "change_type: full_finetune\n"
        f"checks:\n{checks}"
        f"output: {{dir: out, action_log: {tmp_path / 'actions.jsonl'}}}\n"
    )
    return path


def test_an_undeclared_check_name_is_not_built(tmp_path, capsys):
    path = _config(tmp_path, "  unheard-of: {}\n")
    assert main(["gate", "--config", str(path)]) == ExitCode.NOT_BUILT
    assert "No measurement was made" in capsys.readouterr().err
    assert not (tmp_path / "actions.jsonl").exists()


def test_a_single_check_command_needs_that_check_enabled(tmp_path, capsys):
    path = _config(tmp_path, "  lineage: {}\n")
    assert main(["drift", "--config", str(path)]) == ExitCode.USAGE
    assert "enables no check this command runs" in capsys.readouterr().err


def test_missing_battery_sources_are_an_input_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHANGEGATE_DATA", str(tmp_path / "data"))
    path = _config(tmp_path, "  battery.safety-retention: {}\n")
    assert main(["battery", "--config", str(path)]) == ExitCode.USAGE
    assert "changegate fetch-data" in capsys.readouterr().err


def test_lineage_command_runs_only_lineage(tmp_path, capsys):
    path = _config(tmp_path, "  lineage: {}\n  drift.refusal: {}\n")
    # the dummy files are not safetensors, so lineage records an error arm: no verdict
    assert main(["lineage", "--config", str(path)]) == ExitCode.NO_VERDICT
    out = capsys.readouterr().out
    assert "lineage: NO VERDICT (no measurement" in out
    assert "disabled, not run: drift.refusal" in out
    assert (tmp_path / "out" / "results.json").is_file()


def test_fetch_data_fetches_every_pinned_source(tmp_path, monkeypatch, capsys):
    got = []
    monkeypatch.setattr(cli, "fetch", lambda pinned, d: got.append((pinned.key, d)) or d)
    assert main(["fetch-data", "--data-dir", str(tmp_path)]) == ExitCode.OK
    assert [k for k, _ in got] == list(cli.SOURCES)
    assert all(d == tmp_path for _, d in got)


@pytest.mark.parametrize("command", ["census", "record", "organism"])
def test_delegated_commands_have_their_own_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert f"changegate {command}" in capsys.readouterr().out


def test_delegated_usage_error_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["census", "sample"])
    assert exc.value.code == 2
