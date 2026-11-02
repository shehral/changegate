"""Lineage check on synthetic Llama-named checkpoints built in-process (no download)."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")

from changegate.artifacts import ArtifactRef  # noqa: E402
from changegate.checks import Check, Verdict  # noqa: E402
from changegate.checks.lineage import LineageCheck, parse_params  # noqa: E402
from changegate.checks.lineage_tensors import block_of, row_chunks, writer_axis  # noqa: E402
from changegate.config import CheckConfig  # noqa: E402
from changegate.digests import canonical_json, sha256_file  # noqa: E402
from changegate.errors import ArtifactError, ConfigError, NotBuiltError  # noqa: E402
from lineage_builders import (  # noqa: E402
    ADDED,
    CONFIG,
    ENV,
    D,
    L,
    V,
    abliterate,
    lora,
    npy,
    nudge,
    save,
    unit,
    weights,
)

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def zoo(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineage")
    base = weights(0)
    r = unit(7)
    per_layer = torch.stack([torch.zeros(D)] + [unit(20 + i) for i in range(L)])
    z = {
        "root": root,
        "base_t": base,
        "r": r,
        "base": save(root / "base", base),
        "benign": save(root / "benign", nudge(base)),
        "abliterated": save(root / "abl", abliterate(base, lambda b: r)),
        "abliterated_bf16": save(
            root / "abl16", abliterate(base, lambda b: r), dtype=torch.bfloat16
        ),
        "band_only": save(root / "band", abliterate(base, lambda b: r, blocks={1, 2}, embed=False)),
        "outside_band": save(
            root / "outside", abliterate(base, lambda b: r, blocks={0, 3}, embed=False)
        ),
        "per_layer_abl": save(
            root / "pl", abliterate(base, lambda b: per_layer[0 if b is None else b + 1])
        ),
        "fake_base": save(root / "fake", weights(99)),
        "dir": npy(root / "r.npy", r),
        "dir_per_layer": npy(root / "rl.npy", per_layer),
    }
    return z


def ref(path, kind="weights"):
    return ArtifactRef.from_path(path, kind)


def run(base, cand, kind="weights", envelope=ENV, *, bind=True, **params):
    # The zoo's directions are bare .npy files; pin them so they are bound to the base (L-3).
    if bind and params.get("refusal_direction") and "refusal_direction_sha256" not in params:
        params["refusal_direction_sha256"] = sha256_file(Path(params["refusal_direction"]))
    cfg = CheckConfig(envelope=envelope, params=params)
    return LineageCheck().run(ref(base), ref(cand, kind), cfg)


def test_satisfies_the_check_protocol():
    check = LineageCheck()
    assert isinstance(check, Check)
    assert (check.name, check.tier, check.rm_ids) == ("lineage", "B", ("RM-13",))


def test_benign_finetune_passes_with_lineage_match(zoo):
    res = run(zoo["base"], zoo["benign"], refusal_direction=str(zoo["dir"]))
    lin = res.evidence["lineage"]
    assert lin["label"] == "match"
    assert all(b["cosine"] > 0.99 for b in lin["blocks"])
    assert 0 < min(b["rel_distance"] for b in lin["blocks"]) < 0.05
    assert res.score < 0.05
    assert res.verdict is Verdict.PASS
    assert "not detected at recall not measured" in res.describe()
    canonical_json(res.model_dump(mode="json"))  # finite, serializable


def test_identity_lists_every_weight_file_with_digest(zoo):
    res = run(zoo["base"], zoo["benign"])
    ident = res.evidence["identity"]
    files = ident["candidate"]["weight_files"]
    assert [f["relpath"] for f in files] == ["model.safetensors"]
    assert len(files[0]["sha256"]) == 64
    assert ident["base_pin"] == "not pinned"
    assert ident["reverified_after_read"] is True


def test_abliteration_fails_and_every_block_shows_removal(zoo):
    res = run(zoo["base"], zoo["abliterated"], refusal_direction=str(zoo["dir"]))
    abl = res.evidence["abliteration"]
    assert res.evidence["lineage"]["label"] == "match"
    assert res.score > 0.999
    assert res.verdict is Verdict.FAIL
    assert all(b["removed"] > 0.999 for b in abl["blocks"])
    assert {w["tensor"] for w in abl["writers"]} >= {"model.embed_tokens.weight"}
    assert abl["band_blocks"] == [1, 2]


def test_bf16_rounding_leaves_the_abliteration_visible(zoo):
    res = run(zoo["base"], zoo["abliterated_bf16"], refusal_direction=str(zoo["dir"]))
    assert res.score > 0.9
    assert res.verdict is Verdict.FAIL


def test_score_reads_the_band_and_reports_every_block(zoo):
    inside = run(zoo["base"], zoo["band_only"], refusal_direction=str(zoo["dir"]))
    outside = run(zoo["base"], zoo["outside_band"], refusal_direction=str(zoo["dir"]))
    assert inside.score > 0.999 and inside.verdict is Verdict.FAIL
    # Removal outside the band leaves the score at 0 but stays visible per block.
    assert outside.score == pytest.approx(0.0, abs=1e-12)
    removed = {b["block"]: b["removed"] for b in outside.evidence["abliteration"]["blocks"]}
    assert removed[0] > 0.999 and removed[3] > 0.999


def test_per_layer_direction_uses_hidden_states_rows(zoo):
    res = run(
        zoo["base"],
        zoo["per_layer_abl"],
        refusal_direction=str(zoo["dir_per_layer"]),
        direction_layer="per-layer",
    )
    abl = res.evidence["abliteration"]
    assert res.score > 0.999
    assert abl["direction"]["unusable_rows"] == [0]
    assert abl["skipped"] == [
        {"tensor": "model.embed_tokens.weight", "reason": "no usable direction row for this block"}
    ]


def test_fake_base_is_unknown_and_never_pass(zoo):
    res = run(zoo["fake_base"], zoo["benign"], refusal_direction=str(zoo["dir"]))
    lin = res.evidence["lineage"]
    assert lin["label"] == "unknown"
    assert "weight cosine" in lin["unknown_reasons"][0]
    assert res.verdict is Verdict.NO_VERDICT
    assert res.threshold_ref is None
    assert res.score is not None  # reported as a measurement only
    assert "NO VERDICT" in res.describe()


def test_base_pin(zoo):
    pinned = ref(zoo["base"]).sha256
    ok = run(zoo["base"], zoo["benign"], base_sha256=pinned)
    assert ok.evidence["identity"]["base_pin"] == "verified"
    bad = run(zoo["base"], zoo["benign"], base_sha256="0" * 64, refusal_direction=str(zoo["dir"]))
    assert bad.evidence["lineage"]["label"] == "unknown"
    assert bad.verdict is Verdict.NO_VERDICT


def test_without_a_direction_there_is_no_score(zoo):
    res = run(zoo["base"], zoo["abliterated"])
    assert res.score is None
    assert res.verdict is Verdict.NO_VERDICT
    assert res.evidence["abliteration"] is None
    assert res.evidence["lineage"]["label"] == "match"


def test_structural_mismatch(zoo, tmp_path):
    cfg = dict(CONFIG, num_hidden_layers=L + 1)
    cand = save(tmp_path / "deeper", weights(0, layers=L + 1), config=cfg)
    res = run(zoo["base"], cand, refusal_direction=str(zoo["dir"]))
    lin = res.evidence["lineage"]
    assert lin["label"] == "mismatch"
    assert any("num_hidden_layers" in r for r in lin["mismatch_reasons"])
    assert lin["only_in_candidate"]
    assert res.verdict is Verdict.NO_VERDICT


def test_tokenizer_fingerprint(zoo, tmp_path):
    base_t = zoo["base_t"]
    templ = run(zoo["base"], save(tmp_path / "templ", nudge(base_t), template="{{ other }}"))
    tok = templ.evidence["fingerprint"]["tokenizer"]
    assert tok["vocab"] == "same" and tok["chat_template"] == "changed"
    assert templ.evidence["lineage"]["label"] == "match"

    other = {f"x{i}": i for i in range(V - 2)}
    vocab = run(zoo["base"], save(tmp_path / "vocab", nudge(base_t), vocab=other))
    assert vocab.evidence["fingerprint"]["tokenizer"]["vocab"] == "different"
    assert vocab.evidence["lineage"]["label"] == "mismatch"


def test_added_token_resizes_embeddings_without_mismatch(zoo, tmp_path):
    t = nudge(zoo["base_t"])
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        t[name] = torch.cat([t[name], torch.zeros(1, D)])
    added = [*ADDED, {"id": V, "content": "<tool>"}]
    cand = save(tmp_path / "added", t, config=dict(CONFIG, vocab_size=V + 1), added=added)
    res = run(zoo["base"], cand, refusal_direction=str(zoo["dir"]))
    lin, fp = res.evidence["lineage"], res.evidence["fingerprint"]
    assert lin["label"] == "match"
    assert lin["resized"] == ["lm_head.weight", "model.embed_tokens.weight"]
    assert fp["tokenizer"]["added_tokens_new"] == 1
    assert "vocab_size" in fp["config"]["changed"]
    assert res.verdict is Verdict.PASS


def test_sharded_checkpoint_matches_unsharded(zoo, tmp_path):
    sharded = save(tmp_path / "sharded", abliterate(zoo["base_t"], lambda b: zoo["r"]), shards=3)
    one = run(zoo["base"], zoo["abliterated"], refusal_direction=str(zoo["dir"]))
    three = run(zoo["base"], sharded, refusal_direction=str(zoo["dir"]))
    assert three.score == pytest.approx(one.score, abs=1e-9)
    assert len(three.evidence["identity"]["candidate"]["weight_files"]) == 3


def test_broken_shard_index_is_refused(zoo, tmp_path):
    cand = save(tmp_path / "broken", zoo["base_t"], shards=2)
    index = cand / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    first = sorted(data["weight_map"])[0]
    data["weight_map"][first] = "model-00002-of-00002.safetensors"
    index.write_text(json.dumps(data))
    with pytest.raises(ArtifactError, match="is not in shard"):
        run(zoo["base"], cand)


def test_pickle_only_weights_are_refused(zoo, tmp_path):
    cand = tmp_path / "pickled"
    cand.mkdir()
    (cand / "config.json").write_text(json.dumps(CONFIG))
    (cand / "pytorch_model.bin").write_bytes(b"\x80\x04not-opened")
    with pytest.raises(ArtifactError, match=r"never opened \(pytorch_model.bin\)"):
        run(zoo["base"], cand)


def test_pickle_files_beside_safetensors_are_listed_not_opened(zoo, tmp_path):
    cand = save(tmp_path / "withbin", nudge(zoo["base_t"]))
    (cand / "training_args.bin").write_bytes(b"\x80\x04not-opened")
    res = run(zoo["base"], cand)
    assert res.evidence["identity"]["candidate"]["pickle_files"] == ["training_args.bin"]


def test_a_renamed_pickle_is_not_a_readable_safetensors_file(zoo, tmp_path):
    cand = tmp_path / "renamed"
    cand.mkdir()
    (cand / "model.safetensors").write_bytes(b"\x80\x04" + b"\x00" * 64)
    with pytest.raises(ArtifactError, match="not a readable safetensors file"):
        run(zoo["base"], cand)


def test_files_that_change_during_the_run_are_caught(zoo, tmp_path):
    cand = save(tmp_path / "moving", nudge(zoo["base_t"]))
    cand_ref = ref(cand)
    (cand / "config.json").write_text(json.dumps(dict(CONFIG, extra=1)))
    with pytest.raises(ArtifactError, match="differs from the recorded digest"):
        LineageCheck().run(ref(zoo["base"]), cand_ref, CheckConfig(envelope=ENV))


@pytest.mark.parametrize(
    "setup,error,match",
    [
        (lambda p: p / "r.pt", ArtifactError, "must be a .npy"),
        ("pickled", ArtifactError, "pickled arrays refused"),
        ("two_d", ConfigError, "needs params.direction_layer"),
        ("wrong_rows", ArtifactError, "'per-layer' needs 5 rows"),
        ("wrong_width", ArtifactError, "hidden_size is 16"),
    ],
)
def test_direction_file_is_validated(zoo, tmp_path, setup, error, match):
    params = {}
    if callable(setup):
        path = setup(tmp_path)
        path.write_bytes(b"\x80\x04")
    elif setup == "pickled":
        path = tmp_path / "p.npy"
        np.save(path, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    elif setup == "two_d":
        path = npy(tmp_path / "t.npy", np.ones((L + 1, D)))
    elif setup == "wrong_rows":
        path = npy(tmp_path / "w.npy", np.ones((L, D)))
        params["direction_layer"] = "per-layer"
    else:
        path = npy(tmp_path / "x.npy", np.ones(D + 1))
    with pytest.raises(error, match=match):
        run(zoo["base"], zoo["benign"], refusal_direction=str(path), **params)


def test_direction_digest_pin(zoo):
    with pytest.raises(ArtifactError, match="does not match refusal_direction_sha256"):
        run(
            zoo["base"],
            zoo["benign"],
            refusal_direction=str(zoo["dir"]),
            refusal_direction_sha256="1" * 64,
        )


def test_single_row_of_a_2d_direction(zoo):
    stacked = npy(zoo["root"] / "stack.npy", torch.stack([unit(1), zoo["r"], unit(2)]))
    res = run(zoo["base"], zoo["abliterated"], refusal_direction=str(stacked), direction_layer=1)
    assert res.score > 0.999


def adapter_pairs(base_t, r_blocks, rank_seed=None):
    """Rank-1 pairs whose update is -r r^T W on o_proj and down_proj (an abliteration LoRA)."""
    pairs = {}
    for name, w in base_t.items():
        if writer_axis(name) == 0 and block_of(name) in r_blocks:
            r = r_blocks[block_of(name)]
            pairs[name.removesuffix(".weight")] = ((r @ w)[None, :], -r[:, None])
    return pairs


def test_abliteration_lora_is_scored_like_merged_weights(zoo, tmp_path):
    base_t, r = zoo["base_t"], zoo["r"]
    adapter = lora(tmp_path / "abl-lora", adapter_pairs(base_t, dict.fromkeys(range(L), r)))
    pinned = ref(zoo["base"]).sha256
    res = run(
        zoo["base"], adapter, kind="adapter", refusal_direction=str(zoo["dir"]), base_sha256=pinned
    )
    lin = res.evidence["lineage"]
    assert lin["label"] == "match"
    assert lin["adapter"]["ranks"] == [1]
    assert lin["adapter"]["targets"] == ["down_proj", "o_proj"]
    assert lin["adapter"]["modules"] == 2 * L
    assert res.score > 0.999
    assert res.verdict is Verdict.FAIL
    assert res.evidence["fingerprint"]["config"] == {"status": "not applicable: adapter"}


def test_benign_lora_passes_and_reports_update_size(zoo, tmp_path):
    g = torch.Generator().manual_seed(5)
    pairs = {
        f"model.layers.{i}.self_attn.q_proj": (
            torch.randn(2, D, generator=g),
            0.01 * torch.randn(D, 2, generator=g),
        )
        for i in range(L)
    }
    adapter = lora(tmp_path / "benign-lora", pairs, alpha=4, r=2)
    pinned = ref(zoo["base"]).sha256
    res = run(
        zoo["base"], adapter, kind="adapter", refusal_direction=str(zoo["dir"]), base_sha256=pinned
    )
    blocks = res.evidence["lineage"]["blocks"]
    assert all(b["rel_distance"] > 0 for b in blocks)
    assert res.score == pytest.approx(0.0, abs=1e-12)  # q_proj does not write the residual
    assert res.verdict is Verdict.PASS


def test_adapter_without_pinned_base_is_unknown(zoo, tmp_path):
    adapter = lora(tmp_path / "unpinned", adapter_pairs(zoo["base_t"], {0: zoo["r"]}))
    res = run(zoo["base"], adapter, kind="adapter", refusal_direction=str(zoo["dir"]))
    assert res.evidence["lineage"]["label"] == "unknown"
    assert "pin params.base_sha256" in res.evidence["lineage"]["unknown_reasons"][0]
    assert res.verdict is Verdict.NO_VERDICT


def test_misfit_lora_is_a_mismatch(zoo, tmp_path):
    bad = {"model.layers.0.self_attn.o_proj": (torch.ones(1, D + 3), torch.ones(D, 1))}
    adapter = lora(tmp_path / "misfit", bad)
    res = run(zoo["base"], adapter, kind="adapter", base_sha256=ref(zoo["base"]).sha256)
    lin = res.evidence["lineage"]
    assert lin["label"] == "mismatch"
    assert lin["adapter"]["misfit"] == ["model.layers.0.self_attn.o_proj.weight"]


def test_modules_to_save_replace_base_tensors(zoo, tmp_path):
    base_t, r = zoo["base_t"], zoo["r"]
    embed = abliterate(base_t, lambda b: r, blocks=set())["model.embed_tokens.weight"]
    adapter = lora(
        tmp_path / "mts", {}, extra={"base_model.model.model.embed_tokens.weight": embed}
    )
    res = run(
        zoo["base"],
        adapter,
        kind="adapter",
        refusal_direction=str(zoo["dir"]),
        base_sha256=ref(zoo["base"]).sha256,
    )
    writers = {w["tensor"]: w["ratio"] for w in res.evidence["abliteration"]["writers"]}
    assert writers["model.embed_tokens.weight"] < 1e-6
    assert res.evidence["lineage"]["adapter"]["modules_to_save"] == ["model.embed_tokens.weight"]


@pytest.mark.parametrize(
    "config_extra,error",
    [({"use_dora": True}, NotBuiltError), ({"peft_type": "IA3"}, NotBuiltError)],
)
def test_unsupported_adapters_are_refused(zoo, tmp_path, config_extra, error):
    adapter = lora(
        tmp_path / "x", adapter_pairs(zoo["base_t"], {0: zoo["r"]}), config_extra=config_extra
    )
    with pytest.raises(error):
        run(zoo["base"], adapter, kind="adapter")


def test_pickled_adapter_is_refused(zoo, tmp_path):
    adapter = lora(tmp_path / "binlora", {}, save_weights=False)
    (adapter / "adapter_model.bin").write_bytes(b"\x80\x04")
    with pytest.raises(ArtifactError, match=r"never opened \(adapter_model.bin\)"):
        run(zoo["base"], adapter, kind="adapter")


def test_quantized_candidate_and_adapter_base_are_refused(zoo):
    with pytest.raises(NotBuiltError):
        run(zoo["base"], zoo["benign"], kind="quantized")
    with pytest.raises(ArtifactError, match="base must be full weights"):
        LineageCheck().run(ref(zoo["base"], "adapter"), ref(zoo["benign"]), CheckConfig())


@pytest.mark.parametrize("params", [{"nope": 1}, {"band": [0.8, 0.2]}, {"direction_layer": -1}])
def test_params_are_validated(params):
    with pytest.raises(ConfigError):
        parse_params(params)


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA refusal")
def test_cuda_request_without_cuda_is_a_config_error(zoo):
    with pytest.raises(ConfigError, match="no CUDA device"):
        run(zoo["base"], zoo["benign"], device="cuda")


def test_small_chunks_give_the_same_numbers(zoo):
    big = run(zoo["base"], zoo["benign"], refusal_direction=str(zoo["dir"]))
    small = run(zoo["base"], zoo["benign"], refusal_direction=str(zoo["dir"]), max_elements=1024)
    assert small.score == pytest.approx(big.score, abs=1e-12)
    assert small.evidence["lineage"]["blocks"] == pytest.approx(big.evidence["lineage"]["blocks"])


def test_row_chunks_cover_the_rows():
    assert list(row_chunks(5, 4, 8)) == [(0, 2), (2, 4), (4, 5)]
    assert list(row_chunks(1, 0, 8)) == [(0, 1)]


def test_real_transformers_checkpoints(tiny_llama_pair, tmp_path):
    base, cand = tiny_llama_pair
    dirs = npy(tmp_path / "d.npy", torch.stack([unit(40 + i) for i in range(3)]))
    res = run(base, cand, refusal_direction=str(dirs), direction_layer="per-layer")
    lin = res.evidence["lineage"]
    assert lin["label"] == "match"
    dist = {b["block"]: b["rel_distance"] for b in lin["blocks"]}
    assert dist[0] == 0.0 and dist[1] > 0
    removed = {b["block"]: b["removed"] for b in res.evidence["abliteration"]["blocks"]}
    assert removed[0] == pytest.approx(0.0, abs=1e-12)
    assert removed[1] != pytest.approx(0.0, abs=1e-6)


@pytest.mark.gpu
def test_cuda_matches_cpu(zoo):
    cpu = run(zoo["base"], zoo["abliterated"], refusal_direction=str(zoo["dir"]))
    gpu = run(zoo["base"], zoo["abliterated"], refusal_direction=str(zoo["dir"]), device="cuda")
    assert gpu.score == pytest.approx(cpu.score, abs=1e-9)
    assert "gpu_seconds" in gpu.cost


def drift_style_direction(path, vector, model_sha256):
    """A stub written with the drift module's own save_direction (vector + metadata sidecar)."""
    directions = pytest.importorskip("changegate.directions")
    summary = {
        "name": "stub",
        "source": "synthetic",
        "license": "cc0-1.0",
        "n": 2,
        "sha256": "e" * 64,
    }
    meta = directions.DirectionMeta(
        kind="refusal",
        model_sha256=model_sha256,
        model_path="stub",
        layer=L // 2,
        num_layers=L,
        position=-1,
        hidden_size=D,
        positive=summary,
        negative=summary,
        chat_template=False,
        chat_template_sha256=None,
        dtype="float32",
        raw_norm=1.0,
        vector_sha256=directions._vector_digest(np.asarray(vector, dtype=np.float32)),
        created="2026-09-27",
    )
    vec = np.asarray(vector, dtype=np.float32)
    return directions.save_direction(directions.Direction(vector=vec, meta=meta), path)


def test_drift_sidecar_ties_the_direction_to_the_base(zoo, tmp_path):
    base_sha = ref(zoo["base"]).sha256
    ok = drift_style_direction(tmp_path / "base-dir", zoo["r"].numpy(), base_sha)
    res = run(zoo["base"], zoo["abliterated"], refusal_direction=str(ok))
    assert res.score > 0.999
    sidecar = res.evidence["abliteration"]["direction"]["sidecar"]
    assert sidecar["model_sha256"] == base_sha and sidecar["layer"] == L // 2

    other = drift_style_direction(tmp_path / "other-dir", zoo["r"].numpy(), "f" * 64)
    with pytest.raises(ArtifactError, match=r"estimated on artifact f+, not on this base"):
        run(zoo["base"], zoo["abliterated"], refusal_direction=str(other))


def test_tampered_drift_vector_is_refused(zoo, tmp_path):
    npy_path = drift_style_direction(tmp_path / "t", zoo["r"].numpy(), ref(zoo["base"]).sha256)
    np.save(npy_path, np.asarray(unit(8), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ArtifactError, match="does not match its metadata"):
        run(zoo["base"], zoo["abliterated"], refusal_direction=str(npy_path))


def test_an_unbound_direction_gives_no_verdict_with_a_reason(zoo):
    res = run(zoo["base"], zoo["abliterated"], refusal_direction=str(zoo["dir"]), bind=False)
    assert res.score is not None and res.score > 0.5  # still measured and reported
    assert res.verdict is Verdict.NO_VERDICT
    assert res.evidence["abliteration"]["binding"].startswith("unbound")
    pinned = run(zoo["base"], zoo["abliterated"], refusal_direction=str(zoo["dir"]))
    assert pinned.verdict is Verdict.FAIL
    assert pinned.evidence["abliteration"]["binding"] == "pinned by refusal_direction_sha256"
