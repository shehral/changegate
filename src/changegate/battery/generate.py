"""Generation backends behind one protocol: transformers (CPU, MPS, CUDA) and vLLM (CUDA).

Both load a local model directory by path, safetensors only, with remote code off, render
each conversation with the model's own chat template, and decode with the explicit
``Decoding`` settings. Only the stop tokens come from the model's generation config; its
sampling defaults (repetition penalty, top-k) are not applied.

torch, transformers and vllm are imported inside the backends, so this module imports
without them.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from changegate.artifacts import ArtifactRef
from changegate.battery import Decoding, Message
from changegate.digests import sha256_bytes
from changegate.errors import ArtifactError

Backend = Literal["transformers", "vllm"]
# vLLM reports usage statistics to its own server unless one of these is set. setdefault keeps
# an operator's explicit choice; the gate never turns reporting on.
_NO_TELEMETRY = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
Dtype = Literal["auto", "bfloat16", "float16", "float32"]


@runtime_checkable
class Generator(Protocol):
    """A loaded local model that answers conversations.

    Attributes:
        model: the hashed local artifact the weights were loaded from.
        template_sha256: SHA-256 of the chat template text that renders conversations.
        settings: backend, device, dtype and library versions, for the run's log row.

    Does not establish: that the served weights equal ``model``; only that they were read
    from its path.
    """

    model: ArtifactRef
    template_sha256: str
    settings: dict[str, str]

    def generate(self, conversations: Sequence[Sequence[Message]], decoding: Decoding) -> list[str]:
        """Return one response per conversation, in order.

        Args:
            conversations: chat messages, each ending with a user turn (or, for a guard
                judge, the assistant turn under review).
            decoding: token budget, temperature, nucleus mass and seed.

        Returns:
            The decoded new text of each response, special tokens removed.

        Does not establish: reproducibility across backends, devices or batch sizes.
        """
        ...

    def close(self) -> None:
        """Release the model's memory.

        Does not establish: that the device memory is back to its level before loading.
        """
        ...


class GeneratorSpec(BaseModel):
    """How to load a model for generation, as written in a battery's gate.yaml params.

    ``device``, ``batch_size`` apply to transformers; ``gpu_memory_utilization`` and
    ``max_model_len`` to vLLM, whose instances can share one GPU when their fractions sum
    below 1.

    Does not establish: that the settings fit the hardware.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend = "transformers"
    device: str = "cpu"
    dtype: Dtype = "auto"
    batch_size: int = Field(8, ge=1)
    gpu_memory_utilization: float = Field(0.3, gt=0, le=0.95)
    max_model_len: int | None = Field(None, ge=16)


def load_generator(ref: ArtifactRef, spec: GeneratorSpec) -> Generator:
    """Load a local model directory with the backend the spec names.

    Args:
        ref: the hashed model directory.
        spec: backend and its settings.

    Returns:
        A loaded generator.

    Raises:
        ArtifactError: if the artifact is not a directory with a chat template.
        ImportError: if the backend's library is not installed.

    Does not establish: that the directory holds the model its name suggests.
    """
    if spec.backend == "vllm":
        return VLLMGenerator(
            ref,
            dtype=spec.dtype,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            max_model_len=spec.max_model_len,
        )
    return TransformersGenerator(
        ref, device=spec.device, dtype=spec.dtype, batch_size=spec.batch_size
    )


def _model_dir(ref: ArtifactRef) -> str:
    if not ref.path.is_dir():
        raise ArtifactError(f"{ref.path}: generation needs a model directory, not a single file")
    return str(ref.path)


def _template_digest(template: Any, ref: ArtifactRef) -> str:
    if not isinstance(template, str) or not template:
        raise ArtifactError(f"{ref.path}: the tokenizer has no chat template")
    return sha256_bytes(template.encode())


def _as_dicts(conversation: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in conversation]


class TransformersGenerator:
    """Batched generation with transformers, left-padded, on one device.

    Sampled decoding seeds torch once per ``generate`` call, so a rerun reproduces only with
    the same batch size, device and library versions.

    Does not establish: equality with vLLM outputs on the same weights.
    """

    def __init__(
        self, ref: ArtifactRef, device: str = "cpu", dtype: Dtype = "auto", batch_size: int = 8
    ) -> None:
        """Load the tokenizer and model from a local directory.

        Args:
            ref: the hashed model directory.
            device: a torch device string (``cpu``, ``mps``, ``cuda``, ``cuda:1``).
            dtype: weight dtype; ``auto`` uses the checkpoint's.
            batch_size: conversations per forward batch.

        Raises:
            ArtifactError: if the artifact is not a directory or has no chat template.

        Does not establish: that the checkpoint's config matches its weights.
        """
        path = _model_dir(ref)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = ref
        self._torch = torch
        self._batch_size = batch_size
        self._tok = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.template_sha256 = _template_digest(self._tok.chat_template, ref)
        self._tok.padding_side = "left"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._device = torch.device(device)
        # SDPA on MPS returns NaN logits for left-padded rows (seen with torch 2.8), which
        # decode as runs of one token; eager attention does not.
        attention = "eager" if self._device.type == "mps" else None
        self._lm = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=dtype if dtype == "auto" else getattr(torch, dtype),
            attn_implementation=attention,
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._lm.to(self._device).eval()
        eos = self._lm.generation_config.eos_token_id
        self._eos = self._tok.eos_token_id if eos is None else eos
        self.settings = {
            "backend": "transformers",
            "device": str(self._device),
            "dtype": str(self._lm.dtype),
            "attention": attention or "library default",
            "batch_size": str(batch_size),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    def generate(self, conversations: Sequence[Sequence[Message]], decoding: Decoding) -> list[str]:
        """Generate one response per conversation, ``batch_size`` at a time.

        Args:
            conversations: chat messages per request.
            decoding: decoding settings.

        Returns:
            Decoded new text per conversation.

        Does not establish: what the protocol method does not.
        """
        from transformers import GenerationConfig

        torch = self._torch
        sampling = {} if decoding.greedy else {"temperature": decoding.temperature}
        if not decoding.greedy:
            sampling.update(top_p=decoding.top_p, top_k=0)
            torch.manual_seed(decoding.seed)
        config = GenerationConfig(
            max_new_tokens=decoding.max_new_tokens,
            do_sample=not decoding.greedy,
            eos_token_id=self._eos,
            pad_token_id=self._tok.pad_token_id,
            **sampling,
        )
        out: list[str] = []
        for start in range(0, len(conversations), self._batch_size):
            batch = conversations[start : start + self._batch_size]
            texts = [
                self._tok.apply_chat_template(
                    _as_dicts(c), tokenize=False, add_generation_prompt=c[-1].role != "assistant"
                )
                for c in batch
            ]
            enc = self._tok(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                return_token_type_ids=False,
            ).to(self._device)
            with torch.inference_mode():
                ids = self._lm.generate(**enc, generation_config=config, use_model_defaults=False)
            new = ids[:, enc["input_ids"].shape[1] :]
            out.extend(self._tok.batch_decode(new, skip_special_tokens=True))
        return out

    def close(self) -> None:
        """Drop the model and free cached device memory.

        Does not establish: that other references to the model are gone.
        """
        self._lm = None
        gc.collect()
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
        elif self._device.type == "mps":
            self._torch.mps.empty_cache()


def disable_vllm_telemetry() -> None:
    """Set the environment variables that switch off vLLM's usage reporting, if unset.

    Child processes vLLM starts inherit them.

    Does not establish: that vLLM honors them in every version; checked against the
    ``vllm.usage.usage_lib`` variables only.
    """
    for var, value in _NO_TELEMETRY.items():
        os.environ.setdefault(var, value)


class VLLMGenerator:
    """Generation with vLLM's offline ``LLM.chat``, one request per conversation.

    Request ``i`` gets seed ``decoding.seed + i``, so sampled runs reproduce per request on
    the same engine version and hardware.

    Does not establish: equality with transformers outputs on the same weights; comparing the
    two is the served-artifact check's job.
    """

    def __init__(
        self,
        ref: ArtifactRef,
        dtype: Dtype = "auto",
        gpu_memory_utilization: float = 0.3,
        max_model_len: int | None = None,
    ) -> None:
        """Start a vLLM engine on a local model directory.

        Args:
            ref: the hashed model directory.
            dtype: weight dtype; ``auto`` uses the checkpoint's.
            gpu_memory_utilization: fraction of GPU memory this engine may take.
            max_model_len: context length cap; None uses the model's.

        Raises:
            ArtifactError: if the artifact is not a directory or has no chat template.

        vLLM's usage reporting is switched off (``VLLM_NO_USAGE_STATS=1``, ``DO_NOT_TRACK=1``)
        before vllm is imported, unless the environment already sets those variables.

        Does not establish: that vLLM's kernels match the reference implementation, or that
        no other library in the process opens a connection.
        """
        disable_vllm_telemetry()
        import vllm
        from vllm import LLM

        self.model = ref
        self._llm: Any = LLM(
            model=_model_dir(ref),
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            load_format="safetensors",
            trust_remote_code=False,
            seed=0,
        )
        self.template_sha256 = _template_digest(self._llm.get_tokenizer().chat_template, ref)
        self.settings = {
            "backend": "vllm",
            "dtype": dtype,
            "gpu_memory_utilization": str(gpu_memory_utilization),
            "max_model_len": str(max_model_len),
            "vllm": vllm.__version__,
        }

    def generate(self, conversations: Sequence[Sequence[Message]], decoding: Decoding) -> list[str]:
        """Generate all responses in one ``LLM.chat`` call.

        Args:
            conversations: chat messages per request.
            decoding: decoding settings.

        Returns:
            Decoded new text per conversation.

        Raises:
            ValueError: if some conversations end in an assistant turn and others do not.

        Does not establish: what the protocol method does not.
        """
        from vllm import SamplingParams

        params = [
            SamplingParams(
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                max_tokens=decoding.max_new_tokens,
                seed=decoding.seed + i,
            )
            for i in range(len(conversations))
        ]
        tails = {c[-1].role == "assistant" for c in conversations}
        if len(tails) > 1:
            raise ValueError("a batch must not mix conversations ending in assistant and user")
        outputs = self._llm.chat(
            [_as_dicts(c) for c in conversations],
            params,
            use_tqdm=False,
            add_generation_prompt=tails != {True},
        )
        return [o.outputs[0].text for o in outputs]

    def close(self) -> None:
        """Drop the engine and run the garbage collector.

        Does not establish: that vLLM released all GPU memory; its workers may hold some
        until the process exits.
        """
        self._llm = None
        gc.collect()
