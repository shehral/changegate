import pytest

from battery_fakes import refusal_battery
from changegate.artifacts import ArtifactRef
from changegate.battery import (
    Battery,
    BatteryItem,
    BatteryRun,
    Decoding,
    ItemOutcome,
    Message,
    items_digest,
    negative_flips,
)
from changegate.digests import sha256_json

H = "a" * 64


def items():
    return [
        BatteryItem(
            id=f"i{n}",
            messages=(Message(role="user", content=f"q{n}"),),
            category="harmful",
            source="synthetic",
        )
        for n in range(4)
    ]


def run(tmp_path, passed, scorer="rule:refusal-v1", digest=None, decoding=None):
    f = tmp_path / "m.bin"
    f.write_bytes(b"m")
    model = ArtifactRef.from_path(f, "weights")
    outcomes = tuple(
        ItemOutcome(item_id=f"i{n}", passed=p, response_sha256=H) for n, p in enumerate(passed)
    )
    return BatteryRun(
        battery="safety-retention",
        battery_version="0",
        model=model,
        items_sha256=digest or items_digest(items()),
        scorer=scorer,
        outcomes=outcomes,
        decoding=decoding,
    )


def test_items_digest_depends_on_order_and_content():
    a = items()
    assert items_digest(a) == items_digest(items())
    assert items_digest(a) != items_digest(list(reversed(a)))


def test_items_digest_omits_unused_answer_fields():
    item = items()[0]
    legacy = {
        "id": item.id,
        "messages": [{"role": "user", "content": "q0"}],
        "category": "harmful",
        "source": "synthetic",
    }
    assert items_digest([item]) == sha256_json([legacy])
    with_ref = item.model_copy(update={"reference": ("42",)})
    assert items_digest([with_ref]) != items_digest([item])


def test_prompt_is_the_last_user_message():
    item = BatteryItem(
        id="x",
        messages=(
            Message(role="system", content="be terse"),
            Message(role="user", content="first"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="second"),
        ),
        category="c",
        source="synthetic",
    )
    assert item.prompt == "second"
    only_system = BatteryItem(
        id="y", messages=(Message(role="system", content="s"),), category="c", source="synthetic"
    )
    with pytest.raises(ValueError):
        _ = only_system.prompt


def test_negative_flips_counts_only_pass_to_fail(tmp_path):
    base = run(tmp_path, [True, True, False, False])
    cand = run(tmp_path, [True, False, True, False])
    assert negative_flips(base, cand) == ("i1",)


def test_flat_aggregate_can_hide_flips(tmp_path):
    base = run(tmp_path, [True, False, True, False])
    cand = run(tmp_path, [False, True, False, True])
    assert negative_flips(base, cand) == ("i0", "i2")


def test_unscored_items_are_never_flips(tmp_path):
    base = run(tmp_path, [True, None, True, False])
    cand = run(tmp_path, [None, False, False, None])
    assert negative_flips(base, cand) == ("i2",)


@pytest.mark.parametrize(
    "kwargs",
    [{"scorer": "rule:other"}, {"digest": "b" * 64}, {"decoding": Decoding(temperature=1.0)}],
)
def test_incomparable_runs_raise(tmp_path, kwargs):
    base = run(tmp_path, [True] * 4)
    with pytest.raises(ValueError):
        negative_flips(base, run(tmp_path, [True] * 4, **kwargs))


def test_different_item_ids_raise(tmp_path):
    base = run(tmp_path, [True] * 4)
    short = run(tmp_path, [True] * 3)
    with pytest.raises(ValueError):
        negative_flips(base, short)


def test_run_without_the_newer_fields_still_validates(tmp_path):
    r = run(tmp_path, [True, False])
    assert r.decoding is None and r.template_sha256 is None and r.controls == ()
    assert BatteryRun.model_validate_json(r.model_dump_json()) == r


def test_decoding_greedy_flag_and_bounds():
    assert Decoding().greedy
    assert not Decoding(temperature=0.7).greedy
    with pytest.raises(ValueError):
        Decoding(top_p=0)


def test_judged_battery_satisfies_the_protocol():
    assert isinstance(refusal_battery(3), Battery)
