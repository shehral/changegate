import json
import math
import time
from datetime import date

import numpy as np
import pytest

from changegate.artifacts import ArtifactRef
from changegate.checks import Check, Verdict, apply_envelope
from changegate.checks.adapter_diff import (
    CONFIG_FILE,
    PICKLE_FILE,
    WEIGHTS_FILE,
    AdapterDiff,
    WriteMoments,
    read_lora_adapter,
    refusal_band_presence,
    write_factor,
)
from changegate.config import CheckConfig, Envelope
from changegate.digests import canonical_json, sha256_bytes
from changegate.directions import Direction, DirectionMeta, save_direction
from changegate.errors import ArtifactError, ConfigError, NotBuiltError

# numpy on Apple Accelerate raises spurious floating-point flags in matmul (results verified
# against einsum); the check itself runs under np.errstate and tests finiteness explicitly.
pytestmark = [
    pytest.mark.model,
    pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning"),
]
torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

# Shapes of the in-process tiny Llama in conftest.py: (d_out, d_in) per module.
HIDDEN, INTER, KV, LAYERS = 16, 32, 8, 2
TINY = {
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.k_proj": (KV, HIDDEN),
    "self_attn.v_proj": (KV, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HIDDEN),
    "mlp.gate_proj": (INTER, HIDDEN),
    "mlp.up_proj": (INTER, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTER),
}
ENVELOPE = Envelope(warn=2.0, fail=3.0, direction="higher_is_worse")


def key(name, part):
    return f"base_model.model.{name}.lora_{part}.weight"


def lora(rank, seed, shapes=TINY, layers=range(LAYERS), dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer in layers:
        for suffix, (d_out, d_in) in shapes.items():
            name = f"model.layers.{layer}.{suffix}"
            out[key(name, "A")] = (torch.randn(rank, d_in, generator=g) / d_in**0.5).to(dtype)
            out[key(name, "B")] = (torch.randn(d_out, rank, generator=g) / rank**0.5).to(dtype)
    return out


def write_adapter(path, tensors, rank, **config):
    path.mkdir(parents=True, exist_ok=True)
    body = {"peft_type": "LORA", "r": rank, "lora_alpha": 2 * rank} | config
    body.setdefault("base_model_name_or_path", "tiny-llama")
    (path / CONFIG_FILE).write_text(json.dumps(body))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / WEIGHTS_FILE))
    return path


def stub_direction(path, vector, base, kind="refusal", trait=None, layer=1, num_layers=LAYERS):
    vec = np.ascontiguousarray(vector, dtype="<f4")
    meta = DirectionMeta(
        kind=kind,
        trait=trait,
        model_sha256=base.sha256,
        model_path=str(base.path),
        layer=layer,
        num_layers=num_layers,
        position=-1,
        hidden_size=vec.size,
        positive={"stub": "harmful"},
        negative={"stub": "harmless"},
        chat_template=False,
        chat_template_sha256=None,
        dtype="float32",
        raw_norm=float(np.linalg.norm(vec)),
        vector_sha256=sha256_bytes(vec.tobytes()),
        created=date(2026, 9, 27),
    )
    return save_direction(Direction(vector=vec, meta=meta), path)


def unit_of(npy_path):
    v = np.load(npy_path).astype(np.float64)
    return v / np.linalg.norm(v)


def unit(dim, seed):
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"stand-in base weights")
    return ArtifactRef.from_path(root, "weights")


@pytest.fixture
def refusal(tmp_path, base):
    return stub_direction(tmp_path / "card" / "refusal", unit(HIDDEN, 7), base)


def check(base, adapter_dir, refusal_path, envelope=ENVELOPE, **params):
    cfg = CheckConfig(envelope=envelope, params={"refusal": str(refusal_path)} | params)
    return AdapterDiff().run(base, ArtifactRef.from_path(adapter_dir, "adapter"), cfg)


def materialized(tensors, name, scaling):
    a, b = tensors[key(name, "A")].double().numpy(), tensors[key(name, "B")].double().numpy()
    return scaling * b @ a


# --- the math, against materialized updates ---------------------------------------------


def test_write_factor_reproduces_the_update_second_moment():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((5, 11)), rng.standard_normal((9, 5))
    a[4] = a[3]  # rank-deficient A: eigenvalues near zero must not produce NaN
    f = write_factor(a, b, 0.5)
    dw = 0.5 * b @ a
    np.testing.assert_allclose(f @ f.T, dw @ dw.T, atol=1e-10)
    v = unit(9, 1)
    assert np.linalg.norm(f.T @ v) == pytest.approx(np.linalg.norm(v @ dw))


def test_moments_match_brute_force_for_pooled_modules():
    rng = np.random.default_rng(1)
    d = 12
    updates = [(rng.standard_normal((3, 7)), rng.standard_normal((d, 3)), 2.0)]
    updates.append((rng.standard_normal((4, 9)), rng.standard_normal((d, 4)), -0.5))
    v = {"r": unit(d, 2)}
    c = sum((s * b @ a) @ (s * b @ a).T for a, b, s in updates)
    pooled = WriteMoments.from_factors([write_factor(a, b, s) for a, b, s in updates], v)
    second = WriteMoments.from_second_moment(c, v)
    for m in (pooled, second):
        assert m.energy == pytest.approx(np.trace(c))
        assert m.energy_sq == pytest.approx(np.sum(c * c))
        assert m.projected["r"] == pytest.approx(v["r"] @ c @ v["r"])
    eig = np.linalg.eigvalsh(c)
    assert pooled.effective_rank == pytest.approx(eig.sum() ** 2 / (eig**2).sum())


def test_moments_use_the_d_by_d_gram_when_factors_are_wider_than_d():
    rng = np.random.default_rng(3)
    factors = [rng.standard_normal((6, 4)) for _ in range(3)]  # width 12 > d = 6
    c = sum(f @ f.T for f in factors)
    m = WriteMoments.from_factors(factors, {"r": unit(6, 4)})
    assert m.energy_sq == pytest.approx(np.sum(c * c))


@pytest.mark.parametrize("rank", [1, 4, 16])
def test_z_is_standardized_under_the_random_direction_null(rank):
    """Rank-normalized: z has mean 0 and sd 1 over uniform random directions, at every rank."""
    rng = np.random.default_rng(rank)
    d = 64
    f = write_factor(rng.standard_normal((rank, 40)), rng.standard_normal((d, rank)), 1.0)
    directions = {str(i): unit(d, 1000 + i) for i in range(4000)}
    m = WriteMoments.from_factors([f], directions)
    zs = np.array([m.z(k) for k in directions])
    assert abs(zs.mean()) < 0.1
    assert zs.std() == pytest.approx(1.0, abs=0.15)


def test_zero_update_has_no_score():
    m = WriteMoments.from_factors([np.zeros((8, 2))], {"r": unit(8, 0)})
    assert m.z("r") is None and m.effective_rank is None
    assert m.summary()["directions"]["r"] == {
        "projection": 0.0,
        "fraction": None,
        "enrichment": None,
        "z": None,
    }


def test_isotropic_update_has_no_z():
    m = WriteMoments.from_second_moment(np.eye(5), {"r": unit(5, 0)})
    assert m.effective_rank == pytest.approx(5.0)
    assert m.z("r") is None


# --- reading PEFT adapters ------------------------------------------------------------------


def test_reads_peft_names_layers_and_scaling(tmp_path):
    adapter = write_adapter(tmp_path / "a", lora(4, 0), rank=4, lora_alpha=8)
    index = read_lora_adapter(adapter)
    assert len(index.modules) == 2 * len(TINY)
    o = next(m for m in index.modules if m.name == "model.layers.1.self_attn.o_proj")
    assert (o.layer, o.rank, o.d_in, o.d_out, o.scaling, o.writes_residual) == (
        1,
        4,
        HIDDEN,
        HIDDEN,
        2.0,
        True,
    )
    writers = {m.name.rsplit(".", 1)[1] for m in index.modules if m.writes_residual}
    assert writers == {"o_proj", "down_proj"}
    assert [m.layer for m in index.modules] == sorted(m.layer for m in index.modules)


def test_rslora_rank_and_alpha_patterns(tmp_path):
    tensors = lora(4, 0, layers=[0])
    q = "model.layers.0.self_attn.q_proj"
    tensors[key(q, "A")], tensors[key(q, "B")] = torch.randn(2, HIDDEN), torch.randn(HIDDEN, 2)
    adapter = write_adapter(
        tmp_path / "a",
        tensors,
        rank=4,
        lora_alpha=8,
        use_rslora=True,
        rank_pattern={"q_proj": 2},
        alpha_pattern={"^model.layers.0.mlp.down_proj": 1},
    )
    by_name = {m.name: m for m in read_lora_adapter(adapter).modules}
    assert by_name[q].scaling == pytest.approx(8 / math.sqrt(2))
    assert by_name["model.layers.0.mlp.down_proj"].scaling == pytest.approx(1 / 2)
    assert by_name["model.layers.0.self_attn.o_proj"].scaling == pytest.approx(8 / 2)


def test_declared_rank_must_match_the_tensors(tmp_path):
    adapter = write_adapter(tmp_path / "a", lora(4, 0), rank=8)
    with pytest.raises(ArtifactError, match="declares rank 8"):
        read_lora_adapter(adapter)


def test_unpaired_factor_is_an_error(tmp_path):
    tensors = lora(2, 0)
    del tensors[key("model.layers.0.mlp.up_proj", "B")]
    with pytest.raises(ArtifactError, match="no matching pair"):
        read_lora_adapter(write_adapter(tmp_path / "a", tensors, rank=2))


@pytest.mark.parametrize(
    "config,extra,match",
    [
        ({"peft_type": "IA3"}, {}, "only LORA"),
        ({"use_dora": True}, {}, "DoRA"),
        ({"target_parameters": ["mlp.experts.down_proj"]}, {}, "target_parameters"),
        ({}, {"base_model.model.model.layers.0.mlp.down_proj.lora_magnitude_vector": 1}, "DoRA"),
    ],
)
def test_not_built_adapter_types(tmp_path, config, extra, match):
    tensors = lora(2, 0) | {k: torch.ones(HIDDEN) for k in extra}
    with pytest.raises(NotBuiltError, match=match):
        read_lora_adapter(write_adapter(tmp_path / "a", tensors, rank=2, **config))


def test_pickle_adapters_are_not_loaded(tmp_path):
    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / CONFIG_FILE).write_text(json.dumps({"peft_type": "LORA", "r": 2}))
    (adapter / PICKLE_FILE).write_bytes(b"\x80\x04pickle")
    with pytest.raises(ArtifactError, match="pickle"):
        read_lora_adapter(adapter)


def test_malformed_files_are_artifact_errors(tmp_path):
    adapter = write_adapter(tmp_path / "a", lora(2, 0), rank=2)
    (adapter / WEIGHTS_FILE).write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{bad}xxx")
    with pytest.raises(ArtifactError, match="safetensors"):
        read_lora_adapter(adapter)
    (adapter / CONFIG_FILE).write_text("{not json")
    with pytest.raises(ArtifactError):
        read_lora_adapter(adapter)


def test_unanalyzed_tensors_are_listed(tmp_path, base, refusal):
    tensors = lora(2, 0) | {
        "base_model.model.model.layers.0.self_attn.o_proj.lora_B.bias": torch.ones(HIDDEN),
        "base_model.model.lm_head.weight": torch.ones(128, HIDDEN),
    }
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=2), refusal)
    reasons = {u["tensor"]: u["reason"] for u in result.evidence["unanalyzed"]}
    assert "lora_bias" in reasons["base_model.model.model.layers.0.self_attn.o_proj.lora_B.bias"]
    assert "modules_to_save" in reasons["base_model.model.lm_head.weight"]
    assert any("not measured" in n for n in result.evidence["notes"])
    assert result.verdict is Verdict.NO_VERDICT
    assert "lm_head.weight" in result.evidence["verdict_reason"]


def test_unmeasured_residual_writes_never_pass(tmp_path, base, refusal):
    # Review H-3 probe: the only LoRA pair writes orthogonally to r, while a full o_proj copy
    # and a lora_B bias on a writer carry the change; this used to read PASS.
    r = unit_of(refusal)
    g = torch.Generator().manual_seed(3)
    o0 = "model.layers.0.self_attn.o_proj"
    raw_b = torch.randn(HIDDEN, 2, generator=g, dtype=torch.float64).numpy()
    b = raw_b - np.outer(r, r @ raw_b)  # r . B == 0
    weight = torch.zeros(HIDDEN, HIDDEN)
    weight[0] = 100.0
    tensors = {
        key(o0, "A"): torch.randn(2, HIDDEN, generator=g),
        key(o0, "B"): torch.from_numpy(b).float(),
        "base_model.model.model.layers.1.self_attn.o_proj.weight": weight,
        "base_model.model.model.layers.1.self_attn.o_proj.lora_B.bias": torch.full((HIDDEN,), 50.0),
    }
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=2), refusal)
    assert apply_envelope(result.score, ENVELOPE) is Verdict.PASS  # the LoRA pair alone passes
    assert result.verdict is Verdict.NO_VERDICT
    reason = result.evidence["verdict_reason"]
    assert "layers.1.self_attn.o_proj.weight" in reason
    assert "layers.1.self_attn.o_proj.lora_B.bias" in reason


def test_lora_bias_on_a_non_writer_alone_keeps_the_verdict(tmp_path, base, refusal):
    tensors = lora(2, 0) | {
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.bias": torch.ones(KV),
    }
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=2), refusal)
    assert result.verdict is not Verdict.NO_VERDICT
    assert "verdict_reason" not in result.evidence
    assert result.evidence["unanalyzed"]


@pytest.mark.parametrize(
    "pattern,match",
    [
        ("(a+)+$", "quantified group"),
        ("(o_proj|o_pro)*", "quantified group"),
        ("(x){2,}", "quantified group"),
        (r"(a)\1", "backreference"),
        ("a" * 300, "exceeds"),
        ("(", "invalid"),
    ],
)
def test_hostile_rank_and_alpha_patterns_are_refused(tmp_path, pattern, match):
    for field in ("rank_pattern", "alpha_pattern"):
        adapter = write_adapter(tmp_path / field, lora(2, 0), rank=2, **{field: {pattern: 2}})
        with pytest.raises(ArtifactError, match=match):
            read_lora_adapter(adapter)


# --- the check end to end -------------------------------------------------------------------


def test_refusal_aligned_adapter_fails_and_random_adapter_passes(tmp_path, base, refusal):
    r_hat = unit_of(refusal)
    random_adapter = write_adapter(tmp_path / "random", lora(4, 11), rank=4)
    tensors = lora(4, 11)
    for layer in range(LAYERS):  # every writer's first B column points along the direction
        for suffix in ("self_attn.o_proj", "mlp.down_proj"):
            b = tensors[key(f"model.layers.{layer}.{suffix}", "B")]
            b[:, 0] = torch.from_numpy(8 * r_hat).float()
    aligned = write_adapter(tmp_path / "aligned", tensors, rank=4)

    low, high = check(base, random_adapter, refusal), check(base, aligned, refusal)
    assert low.verdict is Verdict.PASS and abs(low.score) < 2.0
    assert high.verdict is Verdict.FAIL and high.score > 10
    assert "FAIL against the user-declared envelope" in high.describe()
    assert "not detected at recall not measured" in low.describe()
    wide = high.evidence["adapter_wide"]["directions"]["refusal"]
    assert wide["z"] == high.score and wide["fraction"] > 0.5
    assert high.rm_ids == ("RM-08", "RM-13", "RM-03") and high.recall is None


def test_projection_magnitudes_match_the_materialized_update(tmp_path, base, refusal):
    tensors = lora(3, 5)
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=3, lora_alpha=6), refusal)
    r_hat = unit_of(refusal)
    rows = {row["module"]: row for row in result.evidence["modules"]}
    writers = [n for n in rows if n.endswith(("o_proj", "down_proj"))]
    total_p = total_e = 0.0
    for name in writers:
        dw = materialized(tensors, name, 2.0)
        p, e = float(np.sum((r_hat @ dw) ** 2)), float(np.sum(dw * dw))
        total_p, total_e = total_p + p, total_e + e
        stats = rows[name]["directions"]["refusal"]
        assert stats["projection"] == pytest.approx(math.sqrt(p), rel=1e-5)
        assert rows[name]["frobenius"] == pytest.approx(math.sqrt(e), rel=1e-5)
        assert stats["enrichment"] == pytest.approx(HIDDEN * p / e, rel=1e-5)
    q = "model.layers.0.self_attn.q_proj"
    assert rows[q]["frobenius"] == pytest.approx(np.linalg.norm(materialized(tensors, q, 2.0)))
    assert "directions" not in rows[q]
    wide = result.evidence["adapter_wide"]
    assert wide["frobenius"] == pytest.approx(math.sqrt(total_e), rel=1e-5)
    assert wide["directions"]["refusal"]["fraction"] == pytest.approx(total_p / total_e, rel=1e-5)


def test_layers_localize_the_write(tmp_path, base, refusal):
    tensors = lora(4, 3)
    b = tensors[key("model.layers.1.mlp.down_proj", "B")]
    b[:, 0] = torch.from_numpy(8 * unit_of(refusal)).float()
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=4), refusal)
    z = {row["layer"]: row["directions"]["refusal"]["z"] for row in result.evidence["layers"]}
    assert z[1] > 5 > z[0]


def test_qv_only_adapter_has_no_verdict(tmp_path, base, refusal):
    shapes = {k: TINY[k] for k in ("self_attn.q_proj", "self_attn.v_proj")}
    adapter = write_adapter(tmp_path / "a", lora(4, 0, shapes=shapes), rank=4)
    result = check(base, adapter, refusal)
    assert result.score is None and result.verdict is Verdict.NO_VERDICT
    assert "v_proj" in result.evidence["notes"][0]
    assert result.evidence["layers"] == []


def test_peft_initialized_adapter_has_no_score(tmp_path, base, refusal):
    tensors = {k: torch.zeros_like(v) if ".lora_B." in k else v for k, v in lora(4, 0).items()}
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=4), refusal)
    assert result.score is None and result.verdict is Verdict.NO_VERDICT


def test_no_envelope_reports_without_verdict(tmp_path, base, refusal):
    result = check(base, write_adapter(tmp_path / "a", lora(2, 0), rank=2), refusal, envelope=None)
    assert result.verdict is Verdict.NO_VERDICT and result.threshold_ref is None
    assert result.score is not None


def test_bfloat16_adapter_matches_float32(tmp_path, base, refusal):
    f32 = check(base, write_adapter(tmp_path / "f32", lora(4, 2), rank=4), refusal)
    bf16 = write_adapter(tmp_path / "bf16", lora(4, 2, dtype=torch.bfloat16), rank=4)
    assert check(base, bf16, refusal).score == pytest.approx(f32.score, abs=0.05)


def test_embedding_lora_is_a_residual_writer(tmp_path, base, refusal):
    rng = torch.Generator().manual_seed(0)
    a, b = torch.randn(4, 128, generator=rng), torch.randn(HIDDEN, 4, generator=rng)
    tensors = {
        "base_model.model.model.embed_tokens.lora_embedding_A": a,
        "base_model.model.model.embed_tokens.lora_embedding_B": b,
    }
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=4, lora_alpha=4), refusal)
    (row,) = result.evidence["modules"]
    assert row["writes_residual"] and row["layer"] is None
    dw = b.double().numpy() @ a.double().numpy()
    expected = np.linalg.norm(unit_of(refusal) @ dw)
    assert row["directions"]["refusal"]["projection"] == pytest.approx(expected, rel=1e-6)


def test_persona_directions_are_reported_not_scored(tmp_path, base, refusal):
    adapter = write_adapter(tmp_path / "a", lora(4, 1), rank=4)
    evil = stub_direction(tmp_path / "card" / "evil", unit(HIDDEN, 8), base, "persona", "evil")
    plain = check(base, adapter, refusal)
    with_persona = check(base, adapter, refusal, personas={"evil": str(evil)})
    assert with_persona.score == plain.score
    wide = with_persona.evidence["adapter_wide"]["directions"]
    assert set(wide) == {"refusal", "persona:evil"}
    assert with_persona.evidence["directions"]["persona:evil"]["trait"] == "evil"


def test_result_is_canonical_json_and_satisfies_the_protocol(tmp_path, base, refusal):
    result = check(base, write_adapter(tmp_path / "a", lora(2, 0), rank=2), refusal)
    canonical_json(result.model_dump(mode="json"))
    assert set(result.cost) == {"wall_seconds", "bytes_read"}
    assert result.evidence["adapter"]["claimed_base"] == "tiny-llama"
    assert result.evidence["base_sha256"] == base.sha256
    assert isinstance(AdapterDiff(), Check) and AdapterDiff().tier == "A"


# --- the refusal-band presence heuristic ----------------------------------------------------


def band_status(tmp_path, base, refusal, shapes=TINY, layers=range(LAYERS), **params):
    adapter = write_adapter(tmp_path / "a", lora(2, 0, shapes=shapes, layers=layers), rank=2)
    return check(base, adapter, refusal, **params).evidence["refusal_band"]


def test_default_band_is_upstream_of_the_direction_layer(tmp_path, base, refusal):
    band = band_status(tmp_path, base, refusal)
    assert band["layers"] == [0] and "upstream" in band["source"]
    assert band["status"] == "writes-in-band"
    assert band["writer_modules_in_band"] == [
        "model.layers.0.mlp.down_proj",
        "model.layers.0.self_attn.o_proj",
    ]
    assert band["band_max_z"] is not None


def test_adapter_outside_the_band(tmp_path, base, refusal):
    assert band_status(tmp_path, base, refusal, layers=[1])["status"] == "outside-band"


def test_adapter_reading_only_in_the_band(tmp_path, base, refusal):
    shapes = {"self_attn.q_proj": TINY["self_attn.q_proj"]}
    band = band_status(tmp_path, base, refusal, shapes=shapes, layers=[0])
    assert band["status"] == "reads-only-in-band" and band["band_max_z"] is None


def test_declared_band_and_undetermined_band(tmp_path, base):
    layer0 = stub_direction(tmp_path / "card" / "r0", unit(HIDDEN, 7), base, layer=0)
    assert band_status(tmp_path, base, layer0)["status"] == "undetermined"
    declared = band_status(tmp_path, base, layer0, refusal_layers=[1])
    assert declared["source"] == "declared" and declared["status"] == "writes-in-band"
    with pytest.raises(ConfigError, match="outside"):
        band_status(tmp_path, base, layer0, refusal_layers=[5])


def test_band_presence_function_directly():
    out = refusal_band_presence([], band=[3], source="declared", layer_z={3: 1.5, 4: 9.0})
    assert out["status"] == "outside-band" and out["band_max_z"] == 1.5


# --- inputs the check refuses ---------------------------------------------------------------


def test_candidate_must_be_an_adapter_directory(tmp_path, base, refusal):
    adapter = write_adapter(tmp_path / "a", lora(2, 0), rank=2)
    cfg = CheckConfig(params={"refusal": str(refusal)})
    with pytest.raises(ArtifactError, match="adapter directory"):
        AdapterDiff().run(base, ArtifactRef.from_path(adapter, "weights"), cfg)


def test_candidate_changed_after_hashing(tmp_path, base, refusal):
    adapter = write_adapter(tmp_path / "a", lora(2, 0), rank=2)
    ref = ArtifactRef.from_path(adapter, "adapter")
    write_adapter(adapter, lora(2, 1), rank=2)
    with pytest.raises(ArtifactError, match="differs"):
        AdapterDiff().run(base, ref, CheckConfig(params={"refusal": str(refusal)}))


def test_direction_from_another_model_is_refused(tmp_path, base):
    other = tmp_path / "other"
    other.mkdir()
    (other / "model.safetensors").write_bytes(b"another model")
    foreign = stub_direction(
        tmp_path / "card" / "r", unit(HIDDEN, 7), ArtifactRef.from_path(other, "weights")
    )
    with pytest.raises(ArtifactError, match="estimated on model"):
        check(base, write_adapter(tmp_path / "a", lora(2, 0), rank=2), foreign)


def test_direction_width_must_match_the_writers(tmp_path, base):
    narrow = stub_direction(tmp_path / "card" / "r", unit(KV, 7), base)
    with pytest.raises(ArtifactError, match="direction width 8"):
        check(base, write_adapter(tmp_path / "a", lora(2, 0), rank=2), narrow)


def test_direction_kinds_are_checked(tmp_path, base, refusal):
    adapter = write_adapter(tmp_path / "a", lora(2, 0), rank=2)
    persona = stub_direction(tmp_path / "card" / "p", unit(HIDDEN, 8), base, "persona", "evil")
    with pytest.raises(ConfigError, match="not refusal"):
        check(base, adapter, persona)
    with pytest.raises(ConfigError, match="not persona 'sycophancy'"):
        check(base, adapter, refusal, personas={"sycophancy": str(persona)})


@pytest.mark.parametrize(
    "params",
    [{}, {"refusal": "r", "surprise": 1}, {"refusal": "r", "writer_modules": ["o_proj("]}],
)
def test_invalid_params_are_config_errors(tmp_path, base, params):
    adapter = ArtifactRef.from_path(write_adapter(tmp_path / "a", lora(2, 0), rank=2), "adapter")
    with pytest.raises(ConfigError):
        AdapterDiff().run(base, adapter, CheckConfig(params=params))


def test_non_finite_tensors_are_refused(tmp_path, base, refusal):
    tensors = lora(2, 0)
    tensors[key("model.layers.0.self_attn.o_proj", "A")][0, 0] = float("nan")
    with pytest.raises(ArtifactError, match="NaN"):
        check(base, write_adapter(tmp_path / "a", tensors, rank=2), refusal)


def test_overflowing_updates_are_errors_not_silent_no_verdicts(tmp_path, base, refusal):
    tensors = {k: torch.full_like(v, 1e38) for k, v in lora(2, 0).items()}
    with pytest.raises(ArtifactError, match="too large"):
        check(base, write_adapter(tmp_path / "a", tensors, rank=2), refusal)


# --- a real tiny model and the 7-9B time budget ---------------------------------------------


def test_adapter_shaped_from_the_tiny_llama_checkpoint(tmp_path, tiny_llama_pair):
    transformers = pytest.importorskip("transformers")
    base_dir, _ = tiny_llama_pair
    model = transformers.AutoModelForCausalLM.from_pretrained(base_dir)
    base = ArtifactRef.from_path(base_dir, "weights")
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for name, module in model.named_modules():
        if name.endswith(("q_proj", "v_proj", "o_proj", "down_proj")):
            tensors[key(name, "A")] = torch.randn(4, module.in_features, generator=g)
            tensors[key(name, "B")] = torch.randn(module.out_features, 4, generator=g)
    hidden = model.config.hidden_size
    refusal = stub_direction(tmp_path / "card" / "r", unit(hidden, 3), base, num_layers=2)
    result = check(base, write_adapter(tmp_path / "a", tensors, rank=4), refusal)
    assert (
        result.evidence["adapter"]["residual_writing_modules"] == 2 * model.config.num_hidden_layers
    )
    assert result.score is not None and math.isfinite(result.score)


# Worst case of Llama-3.1-8B (d 4096, kv 1024), Qwen2.5-7B (MLP 18944, 28 layers) and
# Gemma-2-9B (q 4096, kv 2048, 42 layers), with headroom to 48 layers.
BIG = {
    "self_attn.q_proj": (4096, 4096),
    "self_attn.k_proj": (2048, 4096),
    "self_attn.v_proj": (2048, 4096),
    "self_attn.o_proj": (4096, 4096),
    "mlp.gate_proj": (18944, 4096),
    "mlp.up_proj": (18944, 4096),
    "mlp.down_proj": (4096, 18944),
}
BIG_LAYERS = 48
BUDGET_SECONDS = 15 * 60


def test_rank_64_on_7_to_9b_shapes_fits_the_runner_budget(tmp_path, base):
    """Two full-size layers at rank 64 (fp32), timed from hashing to result, then scaled.

    Every per-module cost is independent of the layer count (the adapter-wide pool adds a
    fixed d x d term per writer), so time scales linearly in layers; the fixed costs are
    scaled too, which overstates. The projection must fit a third of the 15-minute budget,
    leaving room for a slower hosted runner.
    """
    adapter = write_adapter(tmp_path / "big", lora(64, 0, shapes=BIG, layers=range(2)), rank=64)
    refusal = stub_direction(tmp_path / "card" / "r", unit(4096, 1), base, num_layers=BIG_LAYERS)
    started = time.perf_counter()
    cfg = CheckConfig(envelope=ENVELOPE, params={"refusal": str(refusal)})
    result = AdapterDiff().run(base, ArtifactRef.from_path(adapter, "adapter"), cfg)
    projected = (time.perf_counter() - started) * BIG_LAYERS / 2
    assert result.evidence["adapter"]["max_rank"] == 64 and math.isfinite(result.score)
    print(f"adapter-diff: projected {projected:.1f} s for {BIG_LAYERS} layers at rank 64")
    assert projected < BUDGET_SECONDS / 3, f"projected {projected:.0f} s for {BIG_LAYERS} layers"


def test_lineage_refuses_hostile_alpha_patterns():
    from changegate.checks.lineage_adapter import _lora_scale

    cfg = {"lora_alpha": 4, "alpha_pattern": {"(a+)+$": 1}}
    with pytest.raises(ArtifactError, match="quantified group"):
        _lora_scale(cfg, "model.layers.0.self_attn.o_proj.weight", 2)
    assert _lora_scale({"lora_alpha": 4, "alpha_pattern": {"o_proj": 2}}, "x.o_proj.weight", 2) == 1
