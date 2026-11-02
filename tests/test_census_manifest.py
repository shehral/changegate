import pytest

from census_fakes import BASE, BASE_REV, SNAPSHOT, FakeHub, model, rev, world
from changegate.census.hub import HubFile
from changegate.census.manifest import Manifest, discover, fetch_files


@pytest.fixture
def hub():
    return world()


@pytest.fixture
def manifest(hub):
    return discover(hub, BASE, BASE_REV, limit_per_relation=50, snapshot_at=SNAPSHOT)


def test_every_listed_repository_is_an_entry_or_an_exclusion(hub, manifest):
    listed = sum(len(v) for v in hub.listings.values())
    assert len(manifest.entries) + len(manifest.excluded) == listed
    assert len(manifest.entries) == 10


def test_exclusions_state_their_reasons(manifest):
    reasons = {e.repo_id: e.reason for e in manifest.excluded}
    assert reasons == {
        BASE: "the base itself",
        "hank/private-7b": "private or disabled",
        "ivy/gated-7b": "gated (no credentials are used)",
        "jack/norev-7b": "no pinned revision",
        "erin/base-7b-gguf": "GGUF only (the served-artifact arm is not built)",
        "frank/base-7b-pickle": "no safetensors weights (pickle formats are never loaded)",
        "gina/base-7b-nested": "no top-level weight files",
        "user0/base-7b-task0": "listed under an earlier relation",
    }


def test_excluded_repositories_are_never_fetched(hub, manifest):
    fetched = {c[1] for c in hub.calls if c[0] in ("files", "card")}
    assert not fetched & {"hank/private-7b", "ivy/gated-7b", "jack/norev-7b", BASE}
    carded = {c[1] for c in hub.calls if c[0] == "card"}
    assert "frank/base-7b-pickle" not in carded


def test_listing_asks_for_each_relation_with_the_limit(hub, manifest):
    lists = [c for c in hub.calls if c[0] == "list"]
    assert lists == [("list", BASE, "finetune", 50), ("list", BASE, "adapter", 50)]
    assert manifest.relations == ("finetune", "adapter")
    assert manifest.snapshot_at == SNAPSHOT


def test_entries_carry_metadata_declaration_and_purpose(manifest):
    alice = manifest.entry("alice/Base-7B-abliterated")
    assert alice.license == "apache-2.0"
    assert alice.downloads_30d > 0 and alice.last_modified == SNAPSHOT
    assert alice.declaration.declares_safety_removal
    assert alice.declaration.matches[0].source == "name"
    dave = manifest.entry("dave/base-7b-v2")
    assert [m.source for m in dave.declaration.matches] == ["card"]
    assert dave.declared_purpose == "Weights orthogonalized against the refusal direction."
    assert manifest.entry("user1/base-7b-task1").declared_purpose == (
        "A SQL assistant tuned on Spider."
    )
    flags = [e.declaration.declares_safety_removal for e in manifest.entries]
    assert flags.count(True) == 4


def test_adapters_and_full_weights_are_told_apart(manifest):
    full = manifest.entry("user0/base-7b-task0")
    lora = manifest.entry("user4/base-7b-task4")
    assert (full.kind, full.change_type, full.relation) == ("weights", "full_finetune", "finetune")
    assert (lora.kind, lora.change_type, lora.relation) == ("adapter", "adapter_swap", "adapter")
    assert {f.name for f in lora.files} == {"adapter_config.json", "adapter_model.safetensors"}


def test_an_adapter_config_makes_an_adapter_whatever_the_relation():
    hub = FakeHub(
        {"finetune": [model("u/lora", 1)]},
        {"u/lora": {"adapter_config.json": b"{}", "adapter_model.safetensors": b"w"}},
    )
    (entry,) = discover(hub, BASE, BASE_REV, snapshot_at=SNAPSHOT).entries
    assert entry.kind == "adapter"


def test_fetch_selection_skips_subdirectories_code_pickles_and_duplicate_formats():
    files = [
        HubFile(name=n, size=1)
        for n in (
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "consolidated.safetensors",
            "tokenizer.model",
            "chat_template.jinja",
            "modeling_custom.py",
            "pytorch_model.bin",
            "checkpoint-500/model.safetensors",
            "README.md",
        )
    ]
    assert [f.name for f in fetch_files(files)] == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.model",
        "chat_template.jinja",
    ]


def test_skipped_formats_custom_code_and_sizes_are_recorded():
    contents = {
        "config.json": b"{}",
        "model.safetensors": b"abc",
        "pytorch_model.bin": b"p",
        "x.gguf": b"g",
        "modeling_x.py": b"",
    }
    hub = FakeHub({"finetune": [model("u/m", 1)]}, {"u/m": contents})
    (entry,) = discover(hub, BASE, BASE_REV, snapshot_at=SNAPSHOT).entries
    assert entry.skipped_formats == ("bin", "gguf")
    assert entry.has_custom_code
    assert entry.fetch_bytes == 5
    unknown = entry.model_copy(update={"files": (HubFile(name="model.safetensors"),)})
    assert unknown.fetch_bytes is None


def test_manifest_round_trips_with_a_stable_digest(tmp_path, manifest):
    path = tmp_path / "census" / "manifest.json"
    digest = manifest.write(path)
    again = Manifest.load(path)
    assert again == manifest
    assert again.sha256() == digest == manifest.sha256()
    assert manifest.declaration_rules_sha256 == again.declaration_rules_sha256


def test_digest_changes_with_the_snapshot(hub, manifest):
    later = discover(
        hub, BASE, BASE_REV, limit_per_relation=50, snapshot_at=SNAPSHOT.replace(day=3)
    )
    assert later.sha256() != manifest.sha256()


def test_unknown_entry_is_a_key_error(manifest):
    with pytest.raises(KeyError):
        manifest.entry("nobody/none")


def test_revisions_are_pinned(manifest):
    assert manifest.entry("alice/Base-7B-abliterated").revision == rev(1)
