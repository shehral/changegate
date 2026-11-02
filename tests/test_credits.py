"""Every arXiv id cited in code, scripts or docs has a row in CREDITS.md."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARXIV = re.compile(r"(?:arXiv[: ]*|arxiv\.org/abs/)(\d{4}\.\d{4,5})", re.IGNORECASE)
SCANNED = ("src", "scripts", "docs", "examples")
DOCS = ("README.md", "DATA-LICENSES.md", "CONTRIBUTING.md")
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".sh", ".sbatch", ".txt"}


def cited_ids():
    files = [ROOT / d for d in DOCS]
    for top in SCANNED:
        files += [p for p in (ROOT / top).rglob("*") if p.is_file() and p.suffix in SUFFIXES]
    found = {}
    for path in files:
        for match in ARXIV.finditer(path.read_text(errors="replace")):
            found.setdefault(match.group(1), path.relative_to(ROOT))
    return found


def test_every_cited_arxiv_id_is_credited():
    credited = set(ARXIV.findall((ROOT / "CREDITS.md").read_text()))
    missing = {i: str(p) for i, p in cited_ids().items() if i not in credited}
    assert not missing, f"arXiv ids cited without a CREDITS.md row: {missing}"


def test_the_scan_finds_ids():
    assert "2406.11717" in cited_ids()
