import pytest

from changegate.census.declared import (
    PURPOSE_CHARS,
    Declaration,
    DeclarationMatch,
    declared_purpose,
    declares_safety_removal,
    rules_sha256,
    strip_front_matter,
)


@pytest.mark.parametrize(
    ("repo_id", "rule"),
    [
        ("alice/Llama-3-8B-Instruct-abliterated", "abliterat*"),
        ("bob/qwen_7b_Uncensored", "uncensor*"),
        ("carol/model-un-censored", "uncensor*"),
        ("dave/Mistral-7B-DeCensored", "decensor*"),
        ("erin/model-NSFW-v2", "nsfw"),
        ("frank/model-heretic", "heretic"),
        ("gina/jailbroken-7b", "jailbr[eo]*"),
        ("hank/Unfiltered.7B", "unfilter*"),
        ("ivy/base-norefusal", "refusal removal"),
    ],
)
def test_names_that_declare_removal_are_flagged(repo_id, rule):
    d = declares_safety_removal(repo_id, (), "")
    assert d.declares_safety_removal
    assert (d.matches[0].source, d.matches[0].rule) == ("name", rule)


@pytest.mark.parametrize(
    "repo_id",
    [
        "Qwen/Qwen2.5-7B-Instruct",
        "user/sql-coder-7b",
        "user/censorship-detector",
        "user/aligned-7b",
    ],
)
def test_ordinary_names_are_not_flagged(repo_id):
    d = declares_safety_removal(repo_id, ("transformers", "base_model:finetune:Qwen/Q"), "")
    assert not d.declares_safety_removal
    assert d.matches == ()


def test_tags_and_card_body_are_sources():
    card = "---\ntags: [uncensored]\n---\nWe orthogonalized the weights; refusals removed.\n"
    d = declares_safety_removal("user/model", ("not-for-all-audiences",), card)
    got = {(m.source, m.rule) for m in d.matches}
    assert got == {
        ("tag", "not-for-all-audiences"),
        ("card", "orthogonali[sz]*"),
        ("card", "refusal removal"),
    }  # front-matter tags arrive through ``tags``, not the card


def test_one_match_per_source_and_rule():
    d = declares_safety_removal("u/abliterated-abliteration", (), "abliterated abliterated")
    assert [(m.source, m.rule) for m in d.matches] == [
        ("name", "abliterat*"),
        ("card", "abliterat*"),
    ]


def test_negated_mentions_still_match_and_keep_their_text():
    d = declares_safety_removal("u/m", (), "This model is NOT abliterated.")
    assert d.declares_safety_removal
    assert d.matches[0].text == "abliterated"


def test_declaration_flag_must_follow_matches():
    with pytest.raises(ValueError):
        Declaration(declares_safety_removal=True)
    with pytest.raises(ValueError):
        Declaration(
            declares_safety_removal=False,
            matches=(DeclarationMatch(source="name", rule="nsfw", text="nsfw"),),
        )


def test_rules_digest_is_stable():
    assert rules_sha256() == rules_sha256()
    assert len(rules_sha256()) == 64


def test_purpose_is_the_first_prose_paragraph():
    card = (
        "---\nlicense: mit\n---\n"
        "# Title\n\n"
        "[![badge](https://x/y.svg)](https://x)\n"
        "<div>html</div>\n\n"
        "```python\nprint('code first')\n```\n"
        "A **SQL** assistant tuned on [Spider](https://example.org)\n"
        "for `text-to-SQL` tasks.\n\n"
        "Second paragraph.\n"
    )
    assert declared_purpose(card) == "A SQL assistant tuned on Spider for text-to-SQL tasks."


def test_purpose_is_truncated_and_none_without_prose():
    assert len(declared_purpose("x " * 400)) <= PURPOSE_CHARS
    assert declared_purpose("---\na: 1\n---\n# Only a heading\n| a | b |\n") is None
    assert declared_purpose("") is None


def test_front_matter_is_stripped_only_at_the_start():
    assert strip_front_matter("---\na: 1\n---\nbody\n") == "body\n"
    assert strip_front_matter("body\n---\na: 1\n---\n") == "body\n---\na: 1\n---\n"
