"""Generation backends on real (tiny) models. No downloads: the tiny chat model is built
in-process; Qwen2.5-0.5B-Instruct runs only when already in the local cache."""

import shutil

import pytest

from changegate.artifacts import ArtifactRef
from changegate.battery import Decoding, Message
from changegate.battery.batteries import sycophancy
from changegate.battery.generate import GeneratorSpec, TransformersGenerator, load_generator
from changegate.battery.runner import run_battery
from changegate.digests import sha256_bytes
from changegate.errors import ArtifactError

pytestmark = pytest.mark.model

TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)
CONVERSATIONS = [
    (Message(role="user", content="hi"),),
    (Message(role="user", content="a much longer question about something"),),
    (Message(role="system", content="be brief"), Message(role="user", content="and a third")),
]


def _char_tokenizer(template):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    vocab = specials + [chr(c) for c in range(32, 127)] + ["\n"]
    vocab += [f"<x{i}>" for i in range(128 - len(vocab))]
    tok = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({t: i for i, t in enumerate(vocab)}, unk_token="<unk>")
    )
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex(r"[\s\S]"), behavior="isolated"
    )
    tok.decoder = tokenizers.decoders.Fuse()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    fast.chat_template = template
    return fast


@pytest.fixture(scope="module")
def tiny_chat_model(tiny_llama_pair, tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny-chat")
    shutil.copytree(tiny_llama_pair[0], out, dirs_exist_ok=True)
    _char_tokenizer(TEMPLATE).save_pretrained(out)
    return ArtifactRef.from_path(out, "weights")


def test_greedy_generation_is_repeatable_and_batched(tiny_chat_model):
    gen = TransformersGenerator(tiny_chat_model, batch_size=2)
    try:
        first = gen.generate(CONVERSATIONS, Decoding(max_new_tokens=6))
        assert len(first) == 3 and all(isinstance(t, str) for t in first)
        assert gen.generate(CONVERSATIONS, Decoding(max_new_tokens=6)) == first
        assert gen.template_sha256 == sha256_bytes(TEMPLATE.encode())
        assert gen.settings["backend"] == "transformers"
        assert gen.settings["device"] == "cpu"
    finally:
        gen.close()


def test_sampled_generation_repeats_with_its_seed(tiny_chat_model):
    gen = load_generator(tiny_chat_model, GeneratorSpec(batch_size=3))
    try:
        dec = Decoding(max_new_tokens=6, temperature=1.0, seed=7)
        assert gen.generate(CONVERSATIONS, dec) == gen.generate(CONVERSATIONS, dec)
    finally:
        gen.close()


def test_a_model_without_chat_template_is_refused(tiny_llama_pair, tmp_path):
    out = tmp_path / "no-template"
    shutil.copytree(tiny_llama_pair[0], out)
    _char_tokenizer(None).save_pretrained(out)
    with pytest.raises(ArtifactError, match="chat template"):
        TransformersGenerator(ArtifactRef.from_path(out, "weights"))


def test_a_single_file_artifact_is_refused(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF")
    with pytest.raises(ArtifactError, match="directory"):
        TransformersGenerator(ArtifactRef.from_path(f, "quantized"))


def _device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def test_qwen_batched_generation_is_not_degenerate(cached_hf_model):
    """Left-padded batches once decoded as runs of '!' on MPS (NaN logits under SDPA)."""
    ref = ArtifactRef.from_path(cached_hf_model("Qwen/Qwen2.5-0.5B-Instruct"), "weights")
    gen = TransformersGenerator(ref, device=_device(), batch_size=3)
    try:
        out = gen.generate(CONVERSATIONS, Decoding(max_new_tokens=8))
    finally:
        gen.close()
    assert all(text.strip() and len(set(text.strip())) > 1 for text in out), out


def test_qwen_runs_a_battery_end_to_end(cached_hf_model):
    ref = ArtifactRef.from_path(cached_hf_model("Qwen/Qwen2.5-0.5B-Instruct"), "weights")
    battery = sycophancy()
    small = type(battery)(**{**battery.__dict__, "item_list": battery.item_list[:4]})
    gen = TransformersGenerator(ref, device=_device(), batch_size=4)
    try:
        run = run_battery(small, gen, Decoding(max_new_tokens=24))
    finally:
        gen.close()
    assert [o.item_id for o in run.outcomes] == [i.id for i in small.items()]
    assert {o.label for o in run.outcomes} <= {"correct", "incorrect", "both", "neither"}
    assert run.template_sha256 == gen.template_sha256


@pytest.mark.gpu
def test_vllm_generates_on_gpu(cached_hf_model):
    pytest.importorskip("vllm")
    from changegate.battery.generate import VLLMGenerator

    ref = ArtifactRef.from_path(cached_hf_model("Qwen/Qwen2.5-0.5B-Instruct"), "weights")
    gen = VLLMGenerator(ref, gpu_memory_utilization=0.2, max_model_len=1024)
    try:
        out = gen.generate(CONVERSATIONS, Decoding(max_new_tokens=8))
        assert len(out) == 3 and all(t.strip() for t in out)
        sampled = Decoding(max_new_tokens=8, temperature=1.0, seed=1)
        assert gen.generate(CONVERSATIONS, sampled) == gen.generate(CONVERSATIONS, sampled)
    finally:
        gen.close()
