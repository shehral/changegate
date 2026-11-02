import importlib.util
import sys
from pathlib import Path

import pytest

from helpers import HIGHER, result

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "never_claim_lint", ROOT / "tools" / "never_claim_lint.py"
)
ncl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ncl
spec.loader.exec_module(ncl)

SEEDED = [
    ("certify", "We certify this checkpoint."),
    ("certify", "A certified model."),
    ("compliant with", "Compliant with the EU AI Act."),
    ("guarantee", "This guarantees safety."),
    ("ensure", "It ensures nothing regresses."),
    ("prevent", "The gate prevents regressions."),
    ("blocks prompt injection", "It blocks prompt injection."),
    ("secures your agents", "It secures your agents."),
    ("fixes misalignment", "The recipe fixes misalignment."),
    ("automatic repair", "Automatic repair of adapters."),
    ("detects deception", "The probe detects deception."),
    ("'clean' as a verdict", "The candidate came back clean."),
    ("'clean' as a verdict", "A clean bill of health."),
    ("AGI", "Built for AGI."),
    ("superintelligen*", "A superintelligent auditor."),
    ("partner", "Our design partner ran it."),
    ("backed by", "Backed by a big lab."),
    ("book a demo", "Book a demo today."),
    ("pricing", "See pricing."),
    ("waitlist", "Join the waitlist."),
    ("product tier word", "Available in the paid tier."),
    ("'misaligned' as a predicate on a model", "The fine-tune is misaligned."),
    ("'misaligned' as a predicate on a model", "12% of fine-tunes were misaligned."),
    ("retired name", "Plumbwatch v0"),
    ("uncleared name", "pip install missalign"),
]


@pytest.mark.parametrize("rule,text", SEEDED)
def test_each_banned_phrase_is_caught(tmp_path, rule, text):
    p = tmp_path / "seed.md"
    p.write_text(text + "\n")
    hits, _ = ncl.lint([p])
    assert rule in {h.rule for h in hits}


@pytest.mark.parametrize(
    "text",
    [
        "A result under WARN reads not detected at recall r.",
        "This is not an audit, attestation or certification.",
        "Clean, small modules. Run it on a clean offline machine.",
        "Emergent misalignment from narrow fine-tuning (RM-03).",
        "Unpaid reviewers.",
    ],
)
def test_ordinary_prose_passes(tmp_path, text):
    p = tmp_path / "ok.md"
    p.write_text(text + "\n")
    assert ncl.lint([p])[0] == []


def test_quoted_region_is_skipped_and_reported(tmp_path):
    p = tmp_path / "register.md"
    p.write_text(f"intro\n{ncl.QUOTED_ON}\nnever say certified\n{ncl.QUOTED_OFF}\nguarantee\n")
    hits, quoted = ncl.lint([p])
    assert [h.line for h in hits] == [5]
    assert quoted == [(p, 2, 4)]


def test_unterminated_quoted_region_fails(tmp_path):
    p = tmp_path / "register.md"
    p.write_text(f"{ncl.QUOTED_ON}\ncertified\n")
    assert [h.rule for h in ncl.lint([p])[0]] == ["unterminated quoted region"]


def test_python_strings_are_linted_but_docstrings_are_not(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text(
        '"""A module that never says certified."""\n'
        'def f(x):\n    """Does not guarantee anything."""\n'
        '    return f"model is clean: {x}"\n'
    )
    hits, _ = ncl.lint([p])
    assert [(h.rule, h.line) for h in hits] == [("'clean' as a verdict", 4)]


def test_result_wording_passes_the_lint(tmp_path):
    lines = [result(s, HIGHER).describe() for s in (1.0, 2.5, 3.5)] + [result(1.0).describe()]
    p = tmp_path / "out.md"
    p.write_text("\n".join(lines) + "\n")
    assert ncl.lint([p])[0] == []


def test_repository_passes_its_own_lint():
    targets = ncl.default_targets(ROOT)
    assert ROOT / "README.md" in targets
    assert ROOT / "src" / "changegate" / "cli.py" in targets
    hits, _ = ncl.lint(targets)
    assert hits == [], "\n".join(f"{h.path}:{h.line} {h.rule} {h.text!r}" for h in hits)


def test_the_lint_tool_states_its_limits():
    # review L-12: the docstring walk covers the package only, so check the tool here
    import inspect

    for name, obj in vars(ncl).items():
        public = getattr(obj, "__module__", None) == ncl.__name__ and not name.startswith("_")
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and public:
            assert "Does not establish" in (obj.__doc__ or ""), name


def test_the_default_targets_cover_reviews_scripts_and_module_readmes():
    targets = {p.relative_to(ROOT).as_posix() for p in ncl.default_targets()}
    assert "src/changegate/organisms/README.md" in targets
    assert any(t.startswith("scripts/") and t.endswith(".py") for t in targets)
