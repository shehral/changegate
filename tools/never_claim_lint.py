#!/usr/bin/env python3
"""Never-claim lint: fail on banned phrases in user-facing text.

Scans Markdown and YAML docs, and every string literal in ``src/changegate`` and
``scripts`` that is not a docstring (help text, messages, verdict wording). Docstrings are
developer notes and are skipped; TRIAGE.md is an internal escalation log that must quote the
rules it asks about, and is skipped too.

A Markdown region between ``<!-- never-claim: quoted -->`` and ``<!-- never-claim: end -->``
may quote banned phrases (the register itself does). The lint prints every such region so
suppression stays visible.

The lint catches words, not implications. Stdlib only, so CI can run it without installing
the package.

Usage: python3 tools/never_claim_lint.py [PATH ...]   (default: the repository's own targets)
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_GLOBS = (
    "README.md",
    "CREDITS.md",
    "DATA-LICENSES.md",
    "CONTRIBUTING.md",
    "docs/**/*.md",
    "examples/**/*.yaml",
    "src/changegate/**/*.md",
    "scripts/**/*.md",
)
CODE_GLOBS = ("src/changegate/**/*.py", "scripts/**/*.py")
QUOTED_ON = "<!-- never-claim: quoted -->"
QUOTED_OFF = "<!-- never-claim: end -->"

_I = re.IGNORECASE
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("certify", re.compile(r"\bcertif(?:y|ies|ied|ying)\b", _I)),
    ("compliant with", re.compile(r"\bcompliant\s+with\b", _I)),
    ("guarantee", re.compile(r"\bguarantee(?:s|d)?\b", _I)),
    ("ensure", re.compile(r"\bensur(?:e|es|ed|ing)\b", _I)),
    ("prevent", re.compile(r"\bprevent(?:s|ed|ing)?\b", _I)),
    ("blocks prompt injection", re.compile(r"\bblocks?\s+prompt[\s-]+injections?\b", _I)),
    ("secures your agents", re.compile(r"\bsecures?\s+your\s+agents?\b", _I)),
    ("fixes misalignment", re.compile(r"\bfix(?:es|ed|ing)?\s+misalignment\b", _I)),
    ("automatic repair", re.compile(r"\bautomatic(?:ally)?\s+repair", _I)),
    (
        "detects deception",
        re.compile(r"\bdetect(?:s|ed|ing)?\s+deception\b|\bdeception[\s-]+detect", _I),
    ),
    (
        "'clean' as a verdict",
        re.compile(
            r"\b(?:is|are|was|were|comes?\s+back|came\s+back|looks?|found|returned|reported|"
            r"marked|declared|verdict:?)\s+(?:\w+\s+)?clean\b|"
            r"\bclean\s+(?:bill|models?|checkpoints?|fine-?tunes?|adapters?|verdicts?|results?|"
            r"scans?)\b",
            _I,
        ),
    ),
    ("AGI", re.compile(r"\bAGI\b")),
    ("superintelligen*", re.compile(r"superintelligen", _I)),
    ("partner", re.compile(r"\bpartner(?:s|ed|ing|ships?)?\b", _I)),
    ("backed by", re.compile(r"\bbacked\s+by\b", _I)),
    ("book a demo", re.compile(r"\bbook\s+a\s+demo\b", _I)),
    ("pricing", re.compile(r"\bpricing\b", _I)),
    ("waitlist", re.compile(r"\bwait-?list\b", _I)),
    ("product tier word", re.compile(r"\bproduct\s+now\b|\bpaid\b", _I)),
    (
        "'misaligned' as a predicate on a model",
        re.compile(
            r"\b(?:is|are|was|were|be|been|being|becomes?|became|appears?|looks?|seems?)\s+"
            r"(?:\w+\s+)?misaligned\b|\d+(?:\.\d+)?\s*%\s+(?:of\s+)?[\w\s-]{0,40}?\bmisaligned\b",
            _I,
        ),
    ),
    ("retired name", re.compile(r"plumbwatch", _I)),
    ("uncleared name", re.compile(r"\bmiss[\s_-]?align\b", _I)),
)


@dataclass(frozen=True)
class Hit:
    """One banned-phrase match: where it is, which rule, and the matched text.

    Does not establish: that the text makes the claim; the rule matched words.
    """

    path: Path
    line: int
    rule: str
    text: str


def scan_text(text: str, path: Path, first_line: int = 1) -> list[Hit]:
    """Return every rule match in a block of text, with 1-indexed line numbers.

    Does not establish: that text with no match makes no banned claim; the lint catches words,
    not implications.
    """
    hits = []
    for offset, line in enumerate(text.splitlines()):
        for rule, pattern in RULES:
            for m in pattern.finditer(line):
                hits.append(Hit(path, first_line + offset, rule, m.group(0)))
    return hits


def scan_markdown(path: Path) -> tuple[list[Hit], list[tuple[int, int]]]:
    """Scan a doc file, skipping quoted regions. Returns hits and the quoted line ranges.

    Does not establish: anything about the quoted regions, which are not scanned.
    """
    hits, quoted, start = [], [], None
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip() == QUOTED_ON:
            start = n
        elif line.strip() == QUOTED_OFF and start is not None:
            quoted.append((start, n))
            start = None
        elif start is None:
            hits.extend(scan_text(line, path, n))
    if start is not None:
        hits.append(Hit(path, start, "unterminated quoted region", QUOTED_ON))
    return hits, quoted


def scan_python(path: Path) -> list[Hit]:
    """Scan every non-docstring string literal (f-string parts included) in a module.

    Does not establish: anything about docstrings, comments or text built at run time.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            hits.extend(scan_text(node.value, path, node.lineno))
    return hits


def default_targets(root: Path = ROOT) -> list[Path]:
    """List the repository files the lint covers by default.

    Does not establish: that files outside these globs (TRIAGE.md among them) are clean of
    banned phrases.
    """
    found: set[Path] = set()
    for pattern in DOC_GLOBS + CODE_GLOBS:
        found.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(found)


def lint(paths: list[Path]) -> tuple[list[Hit], list[tuple[Path, int, int]]]:
    """Lint files by type. Returns all hits and all quoted (suppressed) regions.

    Does not establish: more than ``scan_python`` and ``scan_markdown`` do.
    """
    hits, quoted = [], []
    for path in paths:
        if path.suffix == ".py":
            hits.extend(scan_python(path))
        else:
            file_hits, regions = scan_markdown(path)
            hits.extend(file_hits)
            quoted.extend((path, a, b) for a, b in regions)
    return hits, quoted


def main(argv: list[str]) -> int:
    """Run the lint and print findings; return 1 on any hit, else 0.

    Does not establish: that 0 hits means the docs make no banned claim.
    """
    paths = [Path(a) for a in argv] or default_targets()
    hits, quoted = lint(paths)
    for path, a, b in quoted:
        print(f"quoted region (not linted): {path}:{a}-{b}")
    for h in hits:
        print(f"{h.path}:{h.line}: banned ({h.rule}): {h.text!r}")
    print(f"never-claim lint: {len(paths)} file(s), {len(hits)} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
