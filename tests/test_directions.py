import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from changegate.battery import BatteryItem, Message
from changegate.digests import sha256_bytes
from changegate.directions import (
    Direction,
    DirectionMeta,
    PersonaSpec,
    PromptSet,
    cosine,
    default_layer,
    difference_of_means,
    dprime,
    is_permissive,
    load_direction,
    load_persona_spec,
    load_prompt_set,
    persona_prompt_sets,
    random_unit_vector,
    save_direction,
    save_prompt_set,
)
from changegate.errors import ArtifactError, ConfigError

ROOT = Path(__file__).resolve().parents[1]


def make_items(n, source="synthetic", prefix="p"):
    return tuple(
        BatteryItem(
            id=f"{prefix}-{i}",
            messages=(Message(role="user", content=f"prompt {i}"),),
            category="harmless",
            source=source,
        )
        for i in range(n)
    )


def make_set(**overrides):
    fields = {"name": "set", "source": "synthetic", "license": "Apache-2.0", "items": make_items(4)}
    return PromptSet(**(fields | overrides))


def make_direction(vector, **overrides):
    vector = np.asarray(vector, dtype=np.float32)
    fields = {
        "kind": "refusal",
        "model_sha256": "a" * 64,
        "model_ref": "org/base@rev",
        "layer": 1,
        "num_layers": 4,
        "position": -1,
        "hidden_size": vector.shape[0],
        "positive": {"name": "h", "n": 2},
        "negative": {"name": "s", "n": 2},
        "chat_template": False,
        "chat_template_sha256": None,
        "dtype": "float32",
        "raw_norm": float(np.linalg.norm(vector)),
        "vector_sha256": sha256_bytes(vector.astype("<f4").tobytes()),
        "created": date(2026, 9, 27),
    }
    return Direction(vector=vector, meta=DirectionMeta(**(fields | overrides)))


@pytest.mark.parametrize(
    "license_id,ok",
    [
        ("MIT", True),
        ("mit", True),
        ("Apache-2.0", True),
        ("CC-BY-4.0", True),
        (" cc0-1.0 ", True),
        ("cc-by-nc-4.0", False),
        ("CC-BY-SA-3.0", False),
        ("gpl-3.0", False),
        ("mit,cc-by-nc-4.0", False),
        ("", False),
    ],
)
def test_is_permissive(license_id, ok):
    assert is_permissive(license_id) is ok


def test_prompt_set_rejects_non_permissive_license():
    with pytest.raises(ValueError, match="permissive"):
        make_set(license="cc-by-nc-4.0")


def test_prompt_set_rejects_mixed_sources_and_duplicate_ids():
    with pytest.raises(ValueError, match="source"):
        make_set(items=make_items(2) + make_items(2, source="advbench", prefix="q"))
    with pytest.raises(ValueError, match="unique"):
        make_set(items=make_items(2) + make_items(2))


def test_prompt_set_digest_tracks_content_and_order():
    a = make_set()
    assert a.sha256 == make_set().sha256
    assert a.sha256 != make_set(items=tuple(reversed(a.items))).sha256
    assert a.summary() == {
        "name": "set",
        "source": "synthetic",
        "license": "Apache-2.0",
        "revision": None,
        "n": 4,
        "sha256": a.sha256,
    }


def test_prompt_set_round_trip(tmp_path):
    original = make_set(revision="abc", seed=0, fetched=date(2026, 9, 27), n_source_rows=10)
    meta_path = save_prompt_set(original, tmp_path / "set.jsonl")
    assert meta_path.name == "set.meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["n"] == 4 and meta["items_sha256"] == original.sha256
    assert load_prompt_set(tmp_path / "set.jsonl") == original


def test_load_prompt_set_detects_edited_rows(tmp_path):
    save_prompt_set(make_set(), tmp_path / "set.jsonl")
    path = tmp_path / "set.jsonl"
    path.write_text(path.read_text().replace("prompt 1", "prompt one"))
    with pytest.raises(ConfigError, match="digest"):
        load_prompt_set(path)


def test_load_prompt_set_detects_dropped_rows(tmp_path):
    save_prompt_set(make_set(), tmp_path / "set.jsonl")
    path = tmp_path / "set.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))
    with pytest.raises(ConfigError):
        load_prompt_set(path)


def test_load_prompt_set_needs_sidecar_and_permissive_license(tmp_path):
    save_prompt_set(make_set(), tmp_path / "set.jsonl")
    meta_path = tmp_path / "set.meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps(meta | {"license": "cc-by-nc-4.0"}))
    with pytest.raises(ConfigError, match="permissive"):
        load_prompt_set(tmp_path / "set.jsonl")
    meta_path.unlink()
    with pytest.raises(ConfigError):
        load_prompt_set(tmp_path / "set.jsonl")


def spec(**overrides):
    fields = {
        "trait": "loudness",
        "source": "synthetic",
        "license": "Apache-2.0",
        "positive_system": ["Shout.", "Be loud."],
        "negative_system": ["Whisper.", "Be quiet.", "Speak softly."],
        "questions": ["Q0?", "Q1?", "Q2?", "Q3?"],
    }
    return PersonaSpec(**(fields | overrides))


def test_persona_prompt_sets_pair_every_system_prompt_with_every_question():
    positive, negative, neutral = persona_prompt_sets(spec())
    assert len(positive.items) == 8 and len(negative.items) == 12 and len(neutral.items) == 4
    # question-major order, so rows reshape to (n_questions, n_system, d)
    assert [i.messages[1].content for i in positive.items[:2]] == ["Q0?", "Q0?"]
    assert [i.messages[0].content for i in positive.items[:2]] == ["Shout.", "Be loud."]
    assert positive.items[0].messages[0].role == "system"
    assert all(len(i.messages) == 1 and i.messages[0].role == "user" for i in neutral.items)
    assert {s.source for s in (positive, negative, neutral)} == {"synthetic"}


def test_persona_spec_license_and_loading(tmp_path):
    with pytest.raises(ValueError, match="permissive"):
        persona_prompt_sets(spec(license="cc-by-nc-4.0"))
    path = tmp_path / "p.json"
    path.write_text(spec().model_dump_json())
    assert load_persona_spec(path) == spec()
    path.write_text("{}")
    with pytest.raises(ConfigError):
        load_persona_spec(path)


def test_difference_of_means_cosine_and_dprime():
    pos = np.array([[2.0, 0.0], [4.0, 0.0]])
    neg = np.array([[0.0, 1.0], [0.0, 3.0]])
    np.testing.assert_allclose(difference_of_means(pos, neg), [3.0, -2.0])
    assert cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 5.0])) == pytest.approx(0.0)
    assert cosine(np.zeros(2), np.ones(2)) is None
    assert dprime(np.array([1.0, 3.0]), np.array([-1.0, -3.0])) == pytest.approx(4 / np.sqrt(2))
    assert dprime(np.array([1.0, 1.0]), np.array([0.0, 0.0])) is None


def test_random_unit_vector_is_a_reproducible_baseline():
    a, b = random_unit_vector(4096, 0), random_unit_vector(4096, 0)
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    fixed = np.ones(4096)
    assert abs(cosine(a, fixed)) < 5 / np.sqrt(4096)
    assert not np.array_equal(a, random_unit_vector(4096, 1))


def test_default_layer():
    assert default_layer(24) == 12
    assert default_layer(1) == 0


def test_direction_round_trip(tmp_path):
    direction = make_direction([3.0, 4.0, 0.0])
    npy = save_direction(direction, tmp_path / "qwen2.5-refusal-L1")
    assert npy.name == "qwen2.5-refusal-L1.npy"
    assert (tmp_path / "qwen2.5-refusal-L1.json").exists()
    for path in (npy, tmp_path / "qwen2.5-refusal-L1.json", tmp_path / "qwen2.5-refusal-L1"):
        loaded = load_direction(path)
        np.testing.assert_array_equal(loaded.vector, direction.vector)
        assert loaded.meta == direction.meta
    np.testing.assert_allclose(loaded.unit, [0.6, 0.8, 0.0])
    assert np.load(npy, allow_pickle=False).dtype == np.dtype("<f4")


def test_load_direction_detects_a_swapped_vector(tmp_path):
    npy = save_direction(make_direction([3.0, 4.0, 0.0]), tmp_path / "d")
    np.save(npy, np.array([3.0, 4.0, 1.0], dtype="<f4"))
    with pytest.raises(ArtifactError, match="does not match"):
        load_direction(npy)
    (tmp_path / "d.json").unlink()
    with pytest.raises(ArtifactError):
        load_direction(npy)


def test_direction_meta_rejects_out_of_range_layer_and_zero_unit():
    with pytest.raises(ValueError, match="num_layers"):
        make_direction([1.0, 0.0], layer=4)
    with pytest.raises(ValueError, match="position"):
        make_direction([1.0, 0.0], position=0)
    with pytest.raises(ValueError, match="zero"):
        _ = make_direction([0.0, 0.0]).unit


@pytest.mark.parametrize("name", ["advbench_harmful", "synthetic_harmless"])
def test_shipped_prompt_sets_verify(name):
    path = ROOT / "data" / "prompts" / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not fetched (scripts/fetch_prompts.py)")
    prompt_set = load_prompt_set(path)
    assert len(prompt_set.items) == 128
    assert is_permissive(prompt_set.license)


def test_shipped_persona_spec_loads():
    positive, negative, neutral = persona_prompt_sets(
        load_persona_spec(ROOT / "data" / "personas" / "sycophancy.json")
    )
    assert len(neutral.items) == 16
    assert len(positive.items) == len(negative.items) == 64


def test_direction_meta_carries_no_local_path_and_drops_the_legacy_key():
    d = make_direction([1.0, 0.0, 0.0])
    assert "model_path" not in d.meta.model_dump()
    legacy = d.meta.model_dump(mode="json") | {"model_path": "/Users/someone/.cache/model"}
    meta = DirectionMeta.model_validate(legacy)
    assert "someone" not in meta.model_dump_json()
