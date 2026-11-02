"""What a repository says about itself: a safety-removal declaration flag and its purpose.

Both are read from the repository's own text with fixed rules; no model reads the card. The
flag is a stratification label for the sampler and a labelled-positive set for the report.
It is kept apart from the analysis: the runner never passes it to the gate, and no gate
result is ever used to set it.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from changegate.digests import sha256_json

Source = Literal["name", "tag", "card"]

# (label, pattern). Patterns run on lowercased text in which "-", "_", "." and "/" are spaces,
# so "Llama_3-abliterated" and "un-censored" match. Bump RULES_VERSION on any change.
RULES_VERSION = "1"
REMOVAL_RULES: tuple[tuple[str, str], ...] = (
    ("abliterat*", r"\babliterat\w*"),
    ("uncensor*", r"\bun ?censor\w*"),
    ("decensor*", r"\bde ?censor\w*"),
    ("nsfw", r"\bnsfw\b"),
    ("not-for-all-audiences", r"\bnot for all audiences\b"),
    ("heretic", r"\bheretic\b"),
    ("jailbr[eo]*", r"\bjailbr[eo]\w*"),
    ("unfilter*", r"\bunfilter\w*"),
    ("unaligned", r"\bunaligned\b"),
    ("orthogonali[sz]*", r"\borthogonali[sz]\w*"),
    ("refusal removal", r"\brefusals? (?:removed|removal|ablat\w*)|\bno ?refusals?\b"),
)
_COMPILED = tuple((label, re.compile(pattern)) for label, pattern in REMOVAL_RULES)
_SEPARATORS = re.compile(r"[-_./]+")
_FRONT_MATTER = re.compile(r"\A﻿?---\s*\n.*?\n---\s*(?:\n|\Z)", re.DOTALL)
_SKIP_LINE = re.compile(r"^(?:#|<|!\[|\||>|-{3,}|={3,}|\[!\[)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"\*\*|__|`")
PURPOSE_CHARS = 280


class DeclarationMatch(BaseModel):
    """One rule that matched: where (name, tag or card), which rule, and the matched text.

    Does not establish: intent. A card that says "not abliterated" matches too; the matched
    text is kept so a reader can audit the label.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Source
    rule: str
    text: str = Field(max_length=64)


class Declaration(BaseModel):
    """The "declares safety removal" flag and the matches behind it.

    Does not establish: that safety behavior was removed, or that an unflagged repository
    kept it. The flag records what the repository's name, tags or card say.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    declares_safety_removal: bool
    matches: tuple[DeclarationMatch, ...] = ()
    rules_version: str = RULES_VERSION

    @model_validator(mode="after")
    def _flag_follows_matches(self) -> Declaration:
        """Require the flag to be set exactly when some rule matched.

        Does not establish: that the rules are the right ones.
        """
        if self.declares_safety_removal != bool(self.matches):
            raise ValueError("declares_safety_removal must be true exactly when a rule matched")
        return self


def rules_sha256() -> str:
    """Return the SHA-256 of the rule list and its version, for manifests.

    Returns:
        64 lowercase hex characters.

    Does not establish: that two manifests with the same digest saw the same card text.
    """
    return sha256_json({"version": RULES_VERSION, "rules": [list(r) for r in REMOVAL_RULES]})


def declares_safety_removal(repo_id: str, tags: tuple[str, ...], card: str) -> Declaration:
    """Apply the removal rules to a repository's name, tags and card body.

    Args:
        repo_id: ``owner/name``; both parts are matched.
        tags: the Hub's tags for the repository.
        card: the raw README.md text; YAML front matter is skipped (its tags arrive in
            ``tags``).

    Returns:
        The flag with every match, at most one per (source, rule).

    Does not establish: anything about the weights; see Declaration.
    """
    texts: tuple[tuple[Source, str], ...] = (
        ("name", repo_id),
        *(("tag", t) for t in tags),
        ("card", strip_front_matter(card)),
    )
    seen: dict[tuple[Source, str], DeclarationMatch] = {}
    for source, raw in texts:
        text = _SEPARATORS.sub(" ", raw.lower())
        for label, pattern in _COMPILED:
            m = pattern.search(text)
            if m and (source, label) not in seen:
                seen[(source, label)] = DeclarationMatch(
                    source=source, rule=label, text=m.group(0)[:64]
                )
    matches = tuple(seen.values())
    return Declaration(declares_safety_removal=bool(matches), matches=matches)


def declared_purpose(card: str) -> str | None:
    """Return the card's first prose paragraph, stripped of markup, at most 280 characters.

    Headings, HTML, images, tables, block quotes, rules and fenced code are skipped; links
    keep their text. No interpretation is applied.

    Args:
        card: the raw README.md text.

    Returns:
        The paragraph, or None when the card has no prose.

    Does not establish: the model's actual use; this is the author's own description, and
    auto-generated card templates are returned as written.
    """
    paragraph: list[str] = []
    in_code = False
    for line in strip_front_matter(card).splitlines():
        s = line.strip()
        if s.startswith(("```", "~~~")):
            in_code = not in_code
            if paragraph:
                break
            continue
        if in_code:
            continue
        if not s or _SKIP_LINE.match(s):
            if paragraph:
                break
            continue
        paragraph.append(s)
    text = _LINK.sub(r"\1", _IMAGE.sub("", " ".join(paragraph)))
    text = " ".join(_EMPHASIS.sub("", text).split())
    return text[:PURPOSE_CHARS].rstrip() or None


def strip_front_matter(card: str) -> str:
    """Remove a leading YAML front-matter block (between ``---`` lines) from card text.

    Args:
        card: the raw README.md text.

    Returns:
        The card body.

    Does not establish: that the front matter was valid YAML.
    """
    return _FRONT_MATTER.sub("", card, count=1)
