"""Every dataset the code fetches or ships has a row in DATA-LICENSES.md before it is used."""

import json
import re
from pathlib import Path

from changegate.battery.sources import SOURCES

ROOT = Path(__file__).resolve().parents[1]


def table_keys():
    text = (ROOT / "DATA-LICENSES.md").read_text()
    return set(re.findall(r"^\| `([a-z][a-z0-9-]*)` \|", text, flags=re.MULTILINE))


def test_every_pinned_battery_source_has_a_row():
    assert set(SOURCES) <= table_keys()


def test_every_shipped_prompt_set_has_a_row():
    for meta in (ROOT / "data" / "prompts").glob("*.meta.json"):
        assert json.loads(meta.read_text())["source"] in table_keys(), meta.name
