"""Fakes for battery tests: a scripted generator, synthetic items and runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from changegate.artifacts import ArtifactRef
from changegate.battery import BatteryItem, BatteryRun, Decoding, ItemOutcome, Message
from changegate.battery.batteries import JudgedBattery
from changegate.battery.judges import RefusalRubric
from changegate.digests import sha256_bytes

REFUSE = "I'm sorry, but I can't help with that."
COMPLY = "Sure, here is how."


class FakeGenerator:
    """Answers each conversation with ``answer(conversation)`` and records every call."""

    def __init__(
        self,
        model: ArtifactRef,
        answer: Callable[[Sequence[Message]], str],
        template: str = "{{ messages }}",
    ) -> None:
        self.model = model
        self.template_sha256 = sha256_bytes(template.encode())
        self.settings = {"backend": "fake"}
        self.calls: list[tuple[list[Sequence[Message]], Decoding]] = []
        self.closed = False
        self._answer = answer

    def generate(self, conversations, decoding):
        self.calls.append((list(conversations), decoding))
        return [self._answer(c) for c in conversations]

    def close(self):
        self.closed = True


def model_ref(tmp_path: Path, name: str, content: bytes | None = None) -> ArtifactRef:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(content or name.encode())
    return ArtifactRef.from_path(d, "weights")


def harmful_items(n: int, category: str = "harmful") -> list[BatteryItem]:
    return [
        BatteryItem(
            id=f"h{i:03d}",
            messages=(Message(role="user", content=f"harmful request {i}"),),
            category=category,
            source="synthetic",
        )
        for i in range(n)
    ]


def refusal_battery(n: int = 20, **kwargs) -> JudgedBattery:
    judge = RefusalRubric()
    return JudgedBattery(
        name="safety-retention",
        version="test",
        rm_ids=("RM-08",),
        item_list=tuple(harmful_items(n)),
        judge=judge,
        pass_labels=frozenset({"refusal"}),
        fail_labels=frozenset({"compliance"}),
        **kwargs,
    )


def run_of(
    model: ArtifactRef,
    passed: Sequence[bool | None],
    *,
    categories: Sequence[str] | None = None,
    decoding: Decoding | None = None,
    controls: tuple[str, ...] = (),
    scorer: str = "rule:test@1",
    battery: str = "safety-retention",
    tag: str = "",
) -> BatteryRun:
    cats = categories or ["harmful"] * len(passed)
    return BatteryRun(
        battery=battery,
        battery_version="test",
        items_sha256="e" * 64,
        model=model,
        scorer=scorer,
        outcomes=tuple(
            ItemOutcome(
                item_id=f"i{n:03d}",
                passed=p,
                response_sha256=sha256_bytes(f"{tag}{n}".encode()),
                label={True: "refusal", False: "compliance", None: "empty"}[p],
                category=c,
            )
            for n, (p, c) in enumerate(zip(passed, cats, strict=True))
        ),
        decoding=decoding or Decoding(),
        controls=controls,
    )
