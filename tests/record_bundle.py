"""A signed record on disk for record tests: artifacts, raw outputs, an action log, keys.

``cryptography`` is a base dependency, so a missing install fails the importing test
module instead of skipping it (review M-5).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

# cryptography is a base dependency: a missing install fails these tests, never skips them.
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from changegate.artifacts import ArtifactRef
from changegate.battery import BatteryItem, BatteryRun, ItemOutcome, Message, items_digest
from changegate.config import Envelope as GateEnvelope
from changegate.config import GateConfig
from changegate.digests import canonical_json, sha256_bytes
from changegate.log import ActionLog
from changegate.record import Identity, NoiseFloor, Record, Repair, Roles
from changegate.record.build import build_statement, seal
from changegate.record.dsse import PAYLOAD_TYPE, add_signature, sign
from changegate.record.keys import key_id
from changegate.record.outputs import RunPair
from changegate.record.verify import Report, verify_record
from helpers import DIGEST, HIGHER, result

BATTERY = "battery.safety"
SAFETY_ENVELOPE = GateEnvelope(warn=0.2, fail=0.5, direction="higher_is_worse")
BASE_PASSED = (True, True, True, True, False, True, True, True)
CANDIDATE_PASSED = (True, True, False, True, False, True, True, True)  # 2/8 fail, 1 flip (i2)
ITEMS = tuple(
    BatteryItem(
        id=f"i{n}",
        messages=(Message(role="user", content=f"synthetic prompt {n}"),),
        category="synthetic",
        source="synthetic",
    )
    for n in range(len(BASE_PASSED))
)
ROLES = Roles(
    requester=Identity(id="ali@example.org"),
    implementer=Identity(id="bob@example.org"),
    approver=Identity(id="carol@example.org"),
)


def fixed_key(seed: int) -> Ed25519PrivateKey:
    """A deterministic test key; never a real one."""
    return Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)


OWNER, VENDOR, OUTSIDE, ATTACKER, AUDITOR = (fixed_key(n) for n in (1, 2, 3, 4, 5))


def kid(key: Ed25519PrivateKey) -> str:
    return key_id(key.public_key())


def pem(key: Ed25519PrivateKey, path: Path, private: bool = False) -> Path:
    s = serialization
    if private:
        body = key.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption())
    else:
        body = key.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo)
    path.write_bytes(body)
    return path


def battery_run(model: ArtifactRef, passed: tuple[bool, ...]) -> BatteryRun:
    return BatteryRun(
        battery="safety",
        battery_version="0",
        items_sha256=items_digest(ITEMS),
        model=model,
        scorer="rule:synthetic-v0",
        outcomes=tuple(
            ItemOutcome(item_id=i.id, passed=ok, response_sha256=sha256_bytes(i.id.encode()))
            for i, ok in zip(ITEMS, passed, strict=True)
        ),
    )


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@dataclass
class Bundle:
    root: Path
    record: Record
    envelope_path: Path
    candidates: list[Path]
    outputs: Path
    log: Path
    trusted: list[Ed25519PrivateKey] = field(default_factory=lambda: [OWNER])
    independent: list[Ed25519PrivateKey] = field(default_factory=list)

    def verify(self, **overrides) -> Report:
        kwargs = dict(
            artifacts=self.candidates,
            outputs_dir=self.outputs,
            log_path=self.log,
            independent_keys=[k.public_key() for k in self.independent],
        )
        kwargs.update(overrides)
        keys = [k.public_key() for k in self.trusted]
        return verify_record(self.envelope_path.read_bytes(), keys, **kwargs)

    def envelope(self) -> dict:
        return json.loads(self.envelope_path.read_text())

    def write_envelope(self, env: dict) -> None:
        self.envelope_path.write_text(json.dumps(env))

    def payload(self) -> dict:
        return json.loads(base64.b64decode(self.envelope()["payload"]))

    def resign(self, data: dict, *keys: Ed25519PrivateKey) -> None:
        self.resign_bytes(canonical_json(data).encode(), *keys)

    def resign_bytes(self, payload: bytes, *keys: Ed25519PrivateKey) -> None:
        keys = keys or (OWNER,)
        env = sign(PAYLOAD_TYPE, payload, keys[0])
        for k in keys[1:]:
            env = add_signature(env, k)
        self.envelope_path.write_text(env.to_json())


def make_bundle(
    root: Path,
    *,
    repair: Repair | None = None,
    verified_by: Identity | None = None,
    countersigned_by: Identity | None = None,
    signers: tuple[Ed25519PrivateKey, ...] = (OWNER,),
    trusted: tuple[Ed25519PrivateKey, ...] | None = None,
    independent: tuple[Ed25519PrivateKey, ...] = (),
    battery_score: float = 0.25,
    metric: str = "fail_rate",
) -> Bundle:
    root.mkdir(parents=True, exist_ok=True)
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic lora weights")
    (adapter / "adapter_config.json").write_text('{"r": 8}')
    gguf = root / "adapter.gguf"
    gguf.write_bytes(b"synthetic quantized build")
    base_file = root / "base.safetensors"
    base_file.write_bytes(b"synthetic base weights")
    base = ArtifactRef.from_path(base_file, "weights")
    cand = ArtifactRef.from_path(adapter, "adapter", base_ref="org/base@rev")
    quant = ArtifactRef.from_path(gguf, "quantized")
    cfg = GateConfig.model_validate(
        {
            "base": {"path": str(base_file), "kind": "weights"},
            "candidate": {"path": str(adapter), "kind": "adapter"},
            "change_type": "adapter_swap",
            "checks": {
                "adapter-diff": {"envelope": HIGHER.model_dump()},
                BATTERY: {"envelope": SAFETY_ENVELOPE.model_dump()},
            },
        }
    )
    log = ActionLog(root / "actions.jsonl")
    log.append("gate.start", config_sha256=DIGEST)
    for check in ("adapter-diff", BATTERY):
        log.append("check.start", check=check)
        log.append("check.end", check=check, status="measured")
    runs = RunPair(battery_run(base, BASE_PASSED), battery_run(cand, CANDIDATE_PASSED))
    record = build_statement(
        cfg,
        base,
        [cand, quant],
        [result(1.0, HIGHER), result(battery_score, SAFETY_ENVELOPE, check=BATTERY)],
        roles=ROLES,
        verified_by=verified_by or Identity(id=kid(OWNER), display="gate operator"),
        runs={BATTERY: runs},
        outputs_dir=root / "outputs",
        noise_floors={BATTERY: NoiseFloor(value=0.0, method="base vs base, same items", runs=1)},
        repair=repair,
        countersigned_by=countersigned_by,
        rollback_to=base,
        log=log,
        metrics={BATTERY: metric},
    )
    env = seal(record, signers[0])
    for k in signers[1:]:
        env = add_signature(env, k)
    envelope_path = root / "record.json"
    envelope_path.write_text(env.to_json())
    return Bundle(
        root=root,
        record=record,
        envelope_path=envelope_path,
        candidates=[adapter, gguf],
        outputs=root / "outputs",
        log=log.path,
        trusted=list(trusted or signers),
        independent=list(independent),
    )


def vendor_repair() -> Repair:
    return Repair(
        recipe_id="resta",
        recipe_sha256=DIGEST,
        target_manifest_sha256="e" * 64,
        supplied_by=Identity(id=kid(VENDOR), display="vendor"),
        applied_by=Identity(id=kid(VENDOR), display="vendor"),
    )
