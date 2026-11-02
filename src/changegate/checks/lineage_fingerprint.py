"""Config and tokenizer fingerprints for the lineage check, read as JSON, never as code.

No tokenizer or config class is built, so nothing an artifact declares in ``auto_map`` runs;
its presence is reported. Only files that are members of the hashed ArtifactRef are read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from changegate.artifacts import ArtifactRef
from changegate.checks.lineage_tensors import read_json
from changegate.digests import sha256_json

_VOLATILE_CONFIG = frozenset({"_name_or_path", "transformers_version", "torch_dtype", "dtype"})
_STRUCTURAL_CONFIG = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
    "moe_intermediate_size",
    "num_experts",
    "num_local_experts",
)
_SPECIAL_TOKENS = ("bos_token", "eos_token", "pad_token", "unk_token")
_VOCAB_FILES = ("vocab.json", "merges.txt", "vocab.txt", "tokenizer.model")


def config_fingerprint(ref: ArtifactRef) -> dict[str, Any] | None:
    """Fingerprint an artifact's ``config.json``, leaving out keys that change on every save.

    Args:
        ref: the artifact.

    Returns:
        The digest of the kept keys, the kept values, and whether ``auto_map`` declares
        repository code (never run here); None when the artifact has no config.

    Does not establish: that equal configs mean equal weights.
    """
    path = member(ref, "config.json")
    if path is None:
        return None
    raw = read_json(path)
    kept = {k: v for k, v in raw.items() if k not in _VOLATILE_CONFIG}
    return {"sha256": sha256_json(kept), "values": kept, "remote_code": "auto_map" in raw}


def compare_configs(base: dict[str, Any] | None, cand: dict[str, Any] | None) -> dict[str, Any]:
    """Compare two config fingerprints.

    Args:
        base: the base's fingerprint.
        cand: the candidate's fingerprint.

    Returns:
        ``status`` (``compared`` or ``unavailable``), both digests, the structural keys that
        differ (architecture sizes; a difference means the models cannot share weights), the
        other keys that differ, and whether either side declares repository code.

    Does not establish: which differences matter for behavior.
    """
    if base is None or cand is None:
        return {"status": "unavailable"}
    b, c = base["values"], cand["values"]
    return {
        "status": "compared",
        "base_sha256": base["sha256"],
        "candidate_sha256": cand["sha256"],
        "structural_mismatch": [k for k in _STRUCTURAL_CONFIG if b.get(k) != c.get(k)],
        "changed": sorted(
            k for k in b.keys() | c.keys() if k not in _STRUCTURAL_CONFIG and b.get(k) != c.get(k)
        ),
        "remote_code_declared": base["remote_code"] or cand["remote_code"],
    }


def tokenizer_fingerprint(ref: ArtifactRef) -> dict[str, Any] | None:
    """Fingerprint an artifact's tokenizer files without loading a tokenizer class.

    Args:
        ref: the artifact.

    Returns:
        The digest of the vocabulary and merges (from ``tokenizer.json``, else from the
        vocabulary files' digests), added tokens by content and id, the chat-template digest,
        special-token strings, and whether ``auto_map`` declares repository code; None when no
        tokenizer file is present.

    Does not establish: that two tokenizers with the same vocabulary normalize text alike.
    """
    out: dict[str, Any] = {}
    if (path := member(ref, "tokenizer.json")) is not None:
        data = read_json(path)
        model = data.get("model") or {}
        out["vocab_sha256"] = sha256_json([model.get("vocab"), model.get("merges")])
        out["added_tokens"] = {
            t["content"]: t["id"]
            for t in data.get("added_tokens") or []
            if isinstance(t, dict) and "content" in t and "id" in t
        }
    elif parts := [[n, d] for n in _VOCAB_FILES if (d := _digest(ref, n)) is not None]:
        out["vocab_sha256"] = sha256_json(parts)
        added = member(ref, "added_tokens.json")
        out["added_tokens"] = read_json(added) if added else {}
    if (path := member(ref, "tokenizer_config.json")) is not None:
        cfg = read_json(path)
        template = cfg.get("chat_template")
        if template is not None:
            out["chat_template_sha256"] = sha256_json(template)
        out["special_tokens"] = {k: _token_text(cfg[k]) for k in _SPECIAL_TOKENS if k in cfg}
        out["remote_code"] = "auto_map" in cfg
    if (digest := _digest(ref, "chat_template.jinja")) is not None:
        out["chat_template_sha256"] = digest
    return out or None


def compare_tokenizers(base: dict[str, Any] | None, cand: dict[str, Any] | None) -> dict[str, Any]:
    """Compare two tokenizer fingerprints.

    Args:
        base: the base's fingerprint.
        cand: the candidate's fingerprint.

    Returns:
        ``vocab`` (``same``, ``different`` or ``unavailable``), the number of added tokens new
        in the candidate, the base's added tokens missing or at another id in the candidate
        (``added_token_conflicts``), ``chat_template`` (``same``, ``changed`` or
        ``unavailable``) and the special tokens that changed.

    Does not establish: that a changed chat template is benign; fine-tunes often change it.
    """
    if base is None or cand is None:
        return {"vocab": "unavailable", "chat_template": "unavailable"}
    added_b, added_c = base.get("added_tokens", {}), cand.get("added_tokens", {})
    special_b, special_c = base.get("special_tokens", {}), cand.get("special_tokens", {})
    return {
        "vocab": _same(base.get("vocab_sha256"), cand.get("vocab_sha256"), "different"),
        "added_tokens_new": len(added_c.keys() - added_b.keys()),
        "added_token_conflicts": sorted(k for k, i in added_b.items() if added_c.get(k) != i),
        "chat_template": _same(
            base.get("chat_template_sha256"), cand.get("chat_template_sha256"), "changed"
        ),
        "special_tokens_changed": sorted(
            k for k in special_b.keys() | special_c.keys() if special_b.get(k) != special_c.get(k)
        ),
        "remote_code_declared": bool(base.get("remote_code") or cand.get("remote_code")),
    }


def member(ref: ArtifactRef, relpath: str) -> Path | None:
    """Return the path of a file inside a directory artifact, if the hashed file list has it.

    Args:
        ref: the artifact.
        relpath: a path relative to the artifact root.

    Returns:
        The path, or None when the artifact is a single file or does not list it.

    Does not establish: that the file is unchanged since hashing; the check re-verifies.
    """
    if ref.path.is_dir() and any(f.relpath == relpath for f in ref.files):
        return ref.path / relpath
    return None


def _digest(ref: ArtifactRef, relpath: str) -> str | None:
    if not ref.path.is_dir():
        return None
    return next((f.sha256 for f in ref.files if f.relpath == relpath), None)


def _same(a: str | None, b: str | None, differ: str) -> str:
    if a is None or b is None:
        return "unavailable"
    return "same" if a == b else differ


def _token_text(token: Any) -> Any:
    return token.get("content") if isinstance(token, dict) else token
