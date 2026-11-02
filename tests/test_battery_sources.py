import io
import urllib.error
import urllib.request

import pytest
from pydantic import ValidationError

from changegate.battery import sources
from changegate.battery.sources import (
    DATA_DIR_ENV,
    SOURCES,
    PinnedFile,
    default_data_dir,
    fetch,
    local_copy,
)
from changegate.digests import sha256_bytes
from changegate.errors import ArtifactError

BODY = b"category,source,forbidden_prompt\nx,custom,y\n"
PERMISSIVE = {"MIT", "Apache-2.0", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0"}


def pinned(body=BODY):
    return PinnedFile(
        key="toy",
        repo="org/repo",
        revision="0" * 40,
        path="data/toy.csv",
        sha256=sha256_bytes(body),
        license="MIT",
    )


def test_shipped_pins_are_well_formed_and_permissive():
    assert set(SOURCES) == {"strongreject", "harmbench", "xstest", "emergent-misalignment"}
    for key, p in SOURCES.items():
        assert p.key == key
        assert p.license in PERMISSIVE
        assert p.url == f"https://raw.githubusercontent.com/{p.repo}/{p.revision}/{p.path}"


@pytest.mark.parametrize(
    "field,value", [("revision", "main"), ("repo", "no slash"), ("sha256", "x"), ("key", "Bad")]
)
def test_pins_reject_floating_or_malformed_fields(field, value):
    data = pinned().model_dump() | {field: value}
    with pytest.raises(ValidationError):
        PinnedFile.model_validate(data)


def test_fetch_downloads_verifies_and_then_reuses(tmp_path, monkeypatch):
    calls = []

    def fake(url, timeout):
        calls.append(url)
        return io.BytesIO(BODY)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    path = fetch(pinned(), tmp_path)
    assert path.read_bytes() == BODY
    assert calls == [pinned().url]
    assert fetch(pinned(), tmp_path) == path
    assert len(calls) == 1
    assert local_copy(pinned(), tmp_path) == path


def test_fetch_rejects_other_bytes_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"tampered"))
    with pytest.raises(ArtifactError, match="do not match"):
        fetch(pinned(), tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_fetch_caps_the_download_size(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "MAX_DOWNLOAD_BYTES", 4)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(BODY))
    with pytest.raises(ArtifactError, match="exceeds"):
        fetch(pinned(), tmp_path)


def test_fetch_wraps_network_errors(tmp_path, monkeypatch):
    def offline(url, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(ArtifactError, match="download"):
        fetch(pinned(), tmp_path)


def test_local_copy_never_downloads_and_checks_the_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", pytest.fail)
    with pytest.raises(ArtifactError, match="missing"):
        local_copy(pinned(), tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(BODY))
    path = fetch(pinned(), tmp_path)
    path.write_bytes(b"edited")
    monkeypatch.setattr(urllib.request, "urlopen", pytest.fail)
    with pytest.raises(ArtifactError, match="does not match"):
        local_copy(pinned(), tmp_path)


def test_default_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert default_data_dir() == tmp_path
    monkeypatch.delenv(DATA_DIR_ENV)
    assert default_data_dir().parts[-2:] == (".changegate", "data")
