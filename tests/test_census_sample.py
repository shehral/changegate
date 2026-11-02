import math

import pytest

from census_fakes import BASE, BASE_REV, SNAPSHOT, world
from changegate.census.manifest import discover
from changegate.census.sample import Sample, draw_sample, rerun_subset


@pytest.fixture
def manifest():
    return discover(world(), BASE, BASE_REV, snapshot_at=SNAPSHOT)


def draw(manifest, seed=7, declared=2, undeclared=3, **kw):
    return draw_sample(
        manifest,
        {"declared": declared, "undeclared": undeclared},
        seed=seed,
        analysis=kw.pop("analysis", "pilot"),
        **kw,
    )


def test_same_seed_same_sample(manifest):
    assert draw(manifest) == draw(manifest)
    assert draw(manifest).sha256() == draw(manifest).sha256()


def test_seed_changes_the_draw(manifest):
    draws = {tuple(i.repo_id for i in draw(manifest, seed=s).items) for s in range(8)}
    assert len(draws) > 1


def test_quotas_are_met_within_strata(manifest):
    s = draw(manifest)
    declared = [i for i in s.items if i.stratum == "declared"]
    undeclared = [i for i in s.items if i.stratum == "undeclared"]
    assert (len(declared), len(undeclared)) == (2, 3)
    for item in declared:
        assert manifest.entry(item.repo_id).declaration.declares_safety_removal
    for item in undeclared:
        assert not manifest.entry(item.repo_id).declaration.declares_safety_removal
        assert item.revision == manifest.entry(item.repo_id).revision
    assert s.population == {"declared": 4, "undeclared": 6}
    assert s.manifest_sha256 == manifest.sha256()


def test_a_short_stratum_is_taken_whole_and_not_topped_up(manifest):
    s = draw(manifest, declared=10, undeclared=1)
    assert sum(i.stratum == "declared" for i in s.items) == 4
    assert sum(i.stratum == "undeclared" for i in s.items) == 1
    assert s.quotas == {"declared": 10, "undeclared": 1}


def test_strata_draws_do_not_depend_on_manifest_order(manifest):
    shuffled = manifest.model_copy(update={"entries": tuple(reversed(manifest.entries))})
    a = {i.repo_id for i in draw(manifest).items}
    b = {i.repo_id for i in draw(shuffled).items}
    assert a == b


def test_confirmatory_needs_a_registration(manifest):
    with pytest.raises(ValueError, match="prereg_id"):
        draw(manifest, analysis="confirmatory")
    s = draw(manifest, analysis="confirmatory", prereg_id="osf.io/abcd1")
    assert (s.analysis, s.prereg_id) == ("confirmatory", "osf.io/abcd1")


@pytest.mark.parametrize("quotas", [{"declared": -1}, {"benign": 2}])
def test_bad_quotas_raise(manifest, quotas):
    with pytest.raises(ValueError):
        draw_sample(manifest, quotas, seed=1, analysis="pilot")


def test_an_empty_draw_is_refused(manifest):
    with pytest.raises(ValueError):
        draw(manifest, declared=0, undeclared=0)


def test_edited_samples_fail_validation(manifest):
    s = draw(manifest)
    data = s.model_dump(mode="json")
    data["items"] = data["items"][:-1]
    with pytest.raises(ValueError):
        Sample.model_validate(data)
    data = s.model_dump(mode="json")
    data["items"].append(data["items"][0])
    data["quotas"]["declared"] += 1
    with pytest.raises(ValueError):
        Sample.model_validate(data)


def test_sample_round_trips(tmp_path, manifest):
    s = draw(manifest)
    digest = s.write(tmp_path / "sample.json")
    assert Sample.load(tmp_path / "sample.json") == s
    assert digest == s.sha256()
    first = s.items[0]
    assert s.stratum_of(first.repo_id) == first.stratum
    with pytest.raises(KeyError):
        s.stratum_of("nobody/none")


def test_rerun_subset_is_a_seeded_ceiling_share(manifest):
    s = draw(manifest, declared=4, undeclared=6)
    picked = rerun_subset(s, 0.1, seed=3)
    assert len(picked) == math.ceil(0.1 * 10) == 1
    assert picked == rerun_subset(s, 0.1, seed=3)
    assert set(rerun_subset(s, 1.0, seed=3)) == {i.repo_id for i in s.items}
    assert list(rerun_subset(s, 0.5, seed=3)) == sorted(rerun_subset(s, 0.5, seed=3))
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            rerun_subset(s, bad, seed=3)
