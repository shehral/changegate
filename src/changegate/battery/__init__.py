"""Tier-D behavior batteries: items, per-item outcomes, runs and negative flips.

A battery is a frozen list of items plus a scorer. Running it against a model yields one
outcome per item; comparing base and candidate runs item by item yields negative flips
(items that passed on the base and fail on the candidate), which an aggregate rate can hide
(Yan et al., arXiv:2011.09161).

This module holds the shared types. The submodules build on them:

- ``sources``: pinned public dataset files, fetched once and verified by SHA-256.
- ``itemsets``: the v0.1 item sets drawn from those files, plus synthetic sycophancy items.
- ``generate``: generation backends (transformers, vLLM) behind one ``Generator`` protocol.
- ``judges``: local scorers (string-match rubrics and local guard models).
- ``batteries``: the four v0.1 batteries.
- ``runner``: generate, score and log one battery on one model.
- ``agreement``: judge-versus-human agreement (Cohen's kappa) from a labels file.
- ``stats``: Wilson, Newcombe and Bonett-Price intervals, and kappa.
- ``report``: flip reports, rate comparisons and the battery's ``CheckResult``.
- ``check``: ``BatteryCheck``, the ``Check`` the gate runs.

The submodules are imported by name (``from changegate.battery.check import BatteryCheck``);
this module imports none of them, so the types load without torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from changegate.artifacts import ArtifactRef
from changegate.checks import RmId
from changegate.digests import sha256_json

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Message(BaseModel):
    """One chat message in a battery item.

    Does not establish: how a given chat template renders it.
    """

    model_config = _FROZEN

    role: Literal["system", "user", "assistant"]
    content: str


class BatteryItem(BaseModel):
    """One prompt in a battery, with the dataset it came from.

    ``source`` is the dataset key as listed in DATA-LICENSES.md, or ``synthetic``.
    ``reference`` and ``distractor`` are used only by answer-matching rubrics: accepted
    spellings of the correct answer, and of the wrong answer a prompt suggests.

    Does not establish: that the item was absent from the candidate's training data.
    """

    model_config = _FROZEN

    id: str = Field(min_length=1)
    messages: tuple[Message, ...] = Field(min_length=1)
    category: str
    source: str = Field(min_length=1)
    reference: tuple[str, ...] = ()
    distractor: tuple[str, ...] = ()

    @property
    def prompt(self) -> str:
        """Return the content of the item's last user message.

        Raises:
            ValueError: if the item has no user message.

        Does not establish: what the model saw; a system message may precede it.
        """
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        raise ValueError(f"item {self.id} has no user message")


class ItemOutcome(BaseModel):
    """The scored outcome of one item on one model.

    ``passed`` is None when the scorer's label is neither a pass nor a fail for the battery
    (an unparsed judge reply, an empty response, an ambiguous answer); such items are left
    out of rates and flips and counted as unscored. ``label`` is the scorer's own label. The
    response itself is kept out of runs; its digest shows two runs saw the same text.

    Does not establish: that the scorer judged the response correctly.
    """

    model_config = _FROZEN

    item_id: str
    passed: bool | None
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: float | None = None
    label: str | None = None
    category: str | None = None


class Decoding(BaseModel):
    """How responses are generated: token budget, temperature, nucleus mass and seed.

    ``temperature == 0`` is greedy decoding. Sampling knobs from a model's own
    ``generation_config`` (repetition penalty, top-k) are not applied, so base and candidate
    are decoded the same way.

    Does not establish: bit-identical outputs across devices, batch sizes or backends.
    """

    model_config = _FROZEN

    max_new_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(0.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    seed: int = Field(0, ge=0)

    @property
    def greedy(self) -> bool:
        """Return True for greedy decoding (temperature 0).

        Does not establish: that a backend's greedy path is deterministic on every device.
        """
        return self.temperature == 0


class BatteryRun(BaseModel):
    """All outcomes of one battery on one model artifact.

    ``scorer`` identifies the scoring instrument: a rule id with version, or the SHA-256 of
    pinned local judge weights together with the digest of the scoring policy text.
    ``template_sha256`` is the digest of the chat template that rendered the prompts, and
    ``controls`` names item categories reported beside the score but not counted in it.

    Does not establish: behavior on prompts outside the battery, formats other than the
    rendered one, or under a different serving configuration.
    """

    model_config = _FROZEN

    battery: str
    battery_version: str
    items_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ArtifactRef
    scorer: str = Field(min_length=1)
    outcomes: tuple[ItemOutcome, ...] = Field(min_length=1)
    decoding: Decoding | None = None
    template_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    controls: tuple[str, ...] = ()


@runtime_checkable
class Battery(Protocol):
    """A frozen set of items and a deterministic scorer for them.

    Attributes:
        name: battery name, as used in gate.yaml check keys (``battery.<name>``).
        version: bumped whenever items or scoring change.
        rm_ids: risk-map modes the battery targets.
        scorer: identity of the scoring instrument (see ``BatteryRun.scorer``).
        decoding: the decoding the battery is defined with.
        control_categories: item categories reported beside the score, not counted in it.
        gives_verdicts: False when results are reported as intervals only.

    Does not establish: that the battery measures its target mode; that needs a known-bad
    library and, for judged batteries, agreement with human labels.
    """

    name: str
    version: str
    rm_ids: tuple[RmId, ...]
    scorer: str
    decoding: Decoding
    control_categories: frozenset[str]
    gives_verdicts: bool

    def items(self) -> Sequence[BatteryItem]:
        """Return the battery's items in a fixed order.

        Does not establish: that the items are representative of any deployment.
        """
        ...

    def score(self, items: Sequence[BatteryItem], responses: Sequence[str]) -> list[ItemOutcome]:
        """Score each response to its item, deterministically and in order.

        Args:
            items: the items that were sent.
            responses: the model's full response texts, one per item.

        Returns:
            One outcome per item, in the same order.

        Does not establish: agreement with a human reader.
        """
        ...


def items_digest(items: Sequence[BatteryItem]) -> str:
    """Return the SHA-256 identifying a battery's exact item list, order included.

    Fields left at their defaults are omitted, so items without answer references hash as
    they did before those fields existed.

    Args:
        items: the battery's items.

    Returns:
        64 lowercase hex characters.

    Does not establish: which scorer was used; that is recorded separately.
    """
    return sha256_json([item.model_dump(mode="json", exclude_defaults=True) for item in items])


def negative_flips(base: BatteryRun, candidate: BatteryRun) -> tuple[str, ...]:
    """List item ids that passed on the base and fail on the candidate.

    Only definite outcomes count: an item unscored on either side is not a flip.

    Args:
        base: the battery run on the base artifact.
        candidate: the same battery run on the candidate.

    Returns:
        Item ids, sorted.

    Raises:
        ValueError: if the two runs used different items, scorers or decoding, or cover
            different ids.

    Does not establish: that a flip is caused by the change rather than sampling noise;
    under non-greedy decoding compare flips with a base-versus-base rerun.
    """
    left = (base.items_sha256, base.scorer, base.decoding)
    right = (candidate.items_sha256, candidate.scorer, candidate.decoding)
    if left != right:
        raise ValueError("runs used different items, scorers or decoding; flips are not comparable")
    before = {o.item_id: o.passed for o in base.outcomes}
    after = {o.item_id: o.passed for o in candidate.outcomes}
    if before.keys() != after.keys():
        raise ValueError("runs cover different item ids")
    return tuple(sorted(i for i, ok in before.items() if ok is True and after[i] is False))
