"""HfHubClient against recorded Hub API responses: no network, real ModelInfo parsing."""

import builtins

import pytest

from census_fakes import FakeHub
from changegate.census.hub import HfHubClient, HubClient, HubModel, hub_model_from_info
from changegate.errors import ChangegateError

hf_api = pytest.importorskip("huggingface_hub.hf_api")
hf_errors = pytest.importorskip("huggingface_hub.errors")

SHA = "0123456789abcdef0123456789abcdef01234567"
LFS = "e" * 64
# Shaped like one item of GET /api/models?filter=base_model:finetune:...&expand[]=...
LISTED = {
    "_id": "66a0",
    "id": "alice/Base-7B-abliterated",
    "author": "alice",
    "sha": SHA,
    "downloads": 4321,
    "downloadsAllTime": 99999,
    "likes": 12,
    "lastModified": "2026-10-30T12:00:00.000Z",
    "createdAt": "2026-09-01T08:00:00.000Z",
    "library_name": "transformers",
    "pipeline_tag": "text-generation",
    "private": False,
    "disabled": False,
    "gated": False,
    "tags": [
        "transformers",
        "safetensors",
        "abliterated",
        "base_model:finetune:base-org/Base-7B-Instruct",
        "license:apache-2.0",
    ],
    "cardData": {
        "license": "other",
        "license_name": "base-community",
        "base_model": "base-org/Base-7B-Instruct",
        "datasets": ["org/sql-set"],
    },
}
# Shaped like GET /api/models/{id}/revision/{sha}?blobs=true (files_metadata=True)
INFO = {
    "id": "alice/Base-7B-abliterated",
    "sha": SHA,
    "siblings": [
        {"rfilename": "config.json", "size": 700, "blobId": "a1"},
        {
            "rfilename": "model.safetensors",
            "size": 14_000_000_000,
            "blobId": "b2",
            "lfs": {"size": 14_000_000_000, "sha256": LFS, "pointerSize": 135},
        },
    ],
}


class FakeApi:
    def __init__(self):
        self.calls = []

    def list_models(self, **kw):
        self.calls.append(("list_models", kw))
        return iter([hf_api.ModelInfo(**LISTED)])

    def model_info(self, repo_id, **kw):
        self.calls.append(("model_info", repo_id, kw))
        return hf_api.ModelInfo(**INFO)

    def hf_hub_download(self, repo_id, filename, **kw):
        self.calls.append(("hf_hub_download", repo_id, filename, kw))
        raise hf_errors.EntryNotFoundError("no README.md")

    def snapshot_download(self, repo_id, **kw):
        self.calls.append(("snapshot_download", repo_id, kw))
        return str(kw["local_dir"])


def test_listing_uses_the_model_tree_filter_and_normalizes():
    api = FakeApi()
    models = list(HfHubClient(api).list_derivatives("base-org/Base-7B-Instruct", "finetune", 50))
    _, kw = api.calls[0]
    assert kw["filter"] == "base_model:finetune:base-org/Base-7B-Instruct"
    assert (kw["sort"], kw["direction"], kw["limit"]) == ("downloads", -1, 50)
    assert "cardData" in kw["expand"] and "full" not in kw
    (m,) = models
    assert isinstance(m, HubModel)
    assert (m.repo_id, m.revision, m.downloads_30d, m.downloads_all_time) == (
        "alice/Base-7B-abliterated",
        SHA,
        4321,
        99999,
    )
    assert m.license == "other:base-community"
    assert m.datasets == ("org/sql-set",)
    assert m.last_modified.year == 2026
    assert "abliterated" in m.tags


def test_license_falls_back_to_the_tag_and_gating_is_boolean():
    raw = {**LISTED, "cardData": None, "gated": "auto"}
    m = hub_model_from_info(hf_api.ModelInfo(**raw))
    assert m.license == "apache-2.0"
    assert m.gated is True
    bare = hub_model_from_info(hf_api.ModelInfo(id="u/m", tags=[]))
    assert (bare.license, bare.revision, bare.downloads_30d) == (None, None, 0)


def test_files_carry_sizes_and_lfs_digests():
    api = FakeApi()
    files = HfHubClient(api).files("alice/Base-7B-abliterated", SHA)
    assert api.calls[0] == (
        "model_info",
        "alice/Base-7B-abliterated",
        {"revision": SHA, "files_metadata": True},
    )
    assert [(f.name, f.size, f.sha256) for f in files] == [
        ("config.json", 700, None),
        ("model.safetensors", 14_000_000_000, LFS),
    ]


def test_missing_card_reads_as_empty():
    assert HfHubClient(FakeApi()).card_text("alice/Base-7B-abliterated", SHA) == ""


def test_card_text_reads_the_downloaded_file(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("---\nlicense: mit\n---\nHello.\n")

    class Api(FakeApi):
        def hf_hub_download(self, repo_id, filename, **kw):
            return str(readme)

    assert HfHubClient(Api()).card_text("u/m", SHA).endswith("Hello.\n")


def test_download_passes_exact_names_into_a_local_dir(tmp_path):
    api = FakeApi()
    dest = tmp_path / "scratch" / "x"
    out = HfHubClient(api).download("u/m", SHA, ["model.safetensors", "config.json"], dest)
    assert out == dest
    _, repo_id, kw = api.calls[0]
    assert repo_id == "u/m"
    assert kw == {
        "revision": SHA,
        "allow_patterns": ["model.safetensors", "config.json"],
        "local_dir": dest,
    }


def test_missing_package_is_a_clear_error(monkeypatch):
    real = builtins.__import__

    def no_hub(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    with pytest.raises(ChangegateError, match="huggingface_hub"):
        HfHubClient()


def test_default_client_sends_no_token(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("huggingface_hub.HfApi", Recorder)
    HfHubClient()
    assert seen == {"token": False}


def test_fakes_satisfy_the_protocol():
    assert isinstance(FakeHub({}, {}), HubClient)
    assert isinstance(HfHubClient(FakeApi()), HubClient)


@pytest.mark.parametrize("bad", ["../etc", "a/../b", "a/b/c", "/abs", "a/.hidden"])
def test_repo_ids_that_could_escape_scratch_are_rejected(bad):
    with pytest.raises(ValueError):
        HubModel(repo_id=bad)
