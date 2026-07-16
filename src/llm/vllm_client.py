"""vLLM client: the single gateway to the LLM.

Responsibilities:
  * Load a model (agent 7B or judge 72B) with the config's dtype/mem settings.
  * Apply the model's chat template to message lists.
  * Generate, returning per-token logprobs (the sampled token's logprob AND the
    top-k alternatives) — the raw material for every uncertainty metric.
  * Support n>1 sampling (self-consistency) and greedy/sampled decoding.

vLLM and torch are imported lazily inside `load()` so this module imports on a
CPU/Windows box for testing the pure-Python parts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def resolve_agent_model(cfg, save: bool = True) -> Dict[str, Any]:
    """Pick the agent model to fit the current GPU, and persist the choice so
    every stage uses the SAME model.

    - GPU < 20 GB (e.g. T4 16 GB)  -> 4-bit AWQ (fits, still gives logprobs)
    - GPU >= 20 GB (L4/A100)       -> fp16 (config default)

    The decision is saved to data_processed/agent_model.json on first call and
    reused thereafter (delete that file to re-decide, e.g. after switching GPU).
    """
    from ..utils.io import save_json, load_json
    path = os.path.join(cfg.path("data_processed"), "agent_model.json")
    if os.path.exists(path):
        return load_json(path)

    vram = None
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass

    base = cfg.models.agent
    if vram is not None and vram < 20:
        # Small GPU: try AWQ quantized variant of the configured model
        awq_name = base.name.rstrip("/") + "-AWQ"
        choice = {"name": awq_name, "dtype": "auto",
                  "gpu_memory_utilization": 0.90, "vram_gb": round(vram, 1),
                  "reason": f"{vram:.0f} GB GPU -> 4-bit AWQ (fits small GPUs)"}
    else:
        choice = {"name": base.name, "dtype": base.dtype,
                  "gpu_memory_utilization": base.gpu_memory_utilization,
                  "vram_gb": round(vram, 1) if vram else None,
                  "reason": f"{('%.0f GB' % vram) if vram else 'unknown GPU'} -> fp16"}
    if save:
        save_json(choice, path)
    return choice


@dataclass
class TokenInfo:
    """Logprob info for one generated token."""
    token_id: int
    token_str: str
    logprob: float                        # logprob of the SAMPLED token
    top_logprobs: Dict[int, float] = field(default_factory=dict)  # token_id -> logprob

    def to_dict(self) -> Dict[str, Any]:
        # top_logprobs keys become strings in JSON; recover with int() on load.
        return {"token_id": self.token_id, "token_str": self.token_str,
                "logprob": self.logprob, "top_logprobs": self.top_logprobs}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TokenInfo":
        return TokenInfo(
            token_id=d["token_id"], token_str=d["token_str"], logprob=d["logprob"],
            top_logprobs={int(k): float(v) for k, v in d.get("top_logprobs", {}).items()},
        )


@dataclass
class GenerationResult:
    """One completion: decoded text + aligned per-token logprob stream."""
    text: str
    token_ids: List[int]
    tokens: List[TokenInfo]

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "token_ids": self.token_ids,
                "tokens": [t.to_dict() for t in self.tokens]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GenerationResult":
        return GenerationResult(
            text=d["text"], token_ids=d.get("token_ids", []),
            tokens=[TokenInfo.from_dict(t) for t in d.get("tokens", [])],
        )


def gpu_vram_gb() -> Optional[float]:
    """Total VRAM of GPU 0 in GB, or None if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return None


def resolve_judge_model(cfg) -> Dict[str, Any]:
    """Pick the annotation judge to fit the GPU.

    >= 60 GB (H100/A100-80)  -> 72B 4-bit AWQ  (best labels)
    >= 34 GB (A100-40)       -> 72B 4-bit AWQ  (tight but fits)
    <  34 GB                 -> 32B 4-bit AWQ  (fallback)
    An explicit `models.judge.fallback.use_fallback: true` always wins.
    """
    jc = cfg.raw["models"]["judge"]
    if jc["fallback"]["use_fallback"]:
        return {"name": jc["fallback"]["name"], "dtype": "auto",
                "gpu_memory_utilization": jc["gpu_memory_utilization"],
                "reason": "forced fallback via config"}
    vram = gpu_vram_gb()
    if vram is None or vram >= 34:
        return {"name": jc["name"], "dtype": jc["dtype"],
                "gpu_memory_utilization": jc["gpu_memory_utilization"],
                "reason": f"{('%.0f GB' % vram) if vram else 'unknown GPU'} -> 72B judge"}
    return {"name": jc["fallback"]["name"], "dtype": "auto",
            "gpu_memory_utilization": jc["gpu_memory_utilization"],
            "reason": f"{vram:.0f} GB -> 72B will not fit, using 32B judge"}


class VLLMClient:
    """Thin wrapper around a vLLM engine with logprob extraction."""

    def __init__(self, model_name: str, dtype: str = "float16",
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.90,
                 logprobs_topk: int = 20, trust_remote_code: bool = True,
                 seed: int = 0):
        self.model_name = model_name
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.logprobs_topk = logprobs_topk
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self._llm = None
        self._tokenizer = None

    # --------------------------------------------------------------------- #
    def load(self) -> "VLLMClient":
        """Instantiate the vLLM engine (lazy import). Call once per session."""
        from vllm import LLM
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code)
        self._llm = LLM(
            model=self.model_name,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            seed=self.seed,
        )
        return self

    # --------------------------------------------------------------------- #
    def _render(self, messages: List[Dict[str, str]]) -> str:
        """Apply the chat template, adding the generation prompt.

        Passes enable_thinking=False for Qwen3 models to suppress
        <think>...</think> blocks that break ReAct action parsing.
        """
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "qwen3" in self.model_name.lower():
            kwargs["enable_thinking"] = False
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _sampling_params(self, temperature: float, max_tokens: int,
                         n: int, stop: Optional[List[str]], seed: Optional[int]):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else 0.95,
            max_tokens=max_tokens,
            n=n,
            logprobs=self.logprobs_topk,   # top-k logprobs per generated token
            stop=stop,
            seed=seed,
        )

    def _to_result(self, comp) -> GenerationResult:
        """Convert one vLLM CompletionOutput to a GenerationResult."""
        tokens: List[TokenInfo] = []
        # comp.logprobs is a list (len = #generated tokens) of
        # {token_id: Logprob(logprob, rank, decoded_token)}
        for pos, tok_id in enumerate(comp.token_ids):
            lp_dict = comp.logprobs[pos] if comp.logprobs else {}
            top = {tid: lp.logprob for tid, lp in lp_dict.items()}
            chosen_lp = top.get(tok_id)
            if chosen_lp is None and lp_dict:
                # sampled token not in returned top-k: use its own entry if present
                chosen_lp = next((lp.logprob for tid, lp in lp_dict.items() if tid == tok_id), None)
            tok_str = self._tokenizer.decode([tok_id]) if self._tokenizer else str(tok_id)
            tokens.append(TokenInfo(
                token_id=int(tok_id),
                token_str=tok_str,
                logprob=float(chosen_lp) if chosen_lp is not None else float("nan"),
                top_logprobs={int(k): float(v) for k, v in top.items()},
            ))
        return GenerationResult(text=comp.text, token_ids=list(comp.token_ids), tokens=tokens)

    # --------------------------------------------------------------------- #
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0,
             max_tokens: int = 512, n: int = 1,
             stop: Optional[List[str]] = None,
             seed: Optional[int] = None) -> List[GenerationResult]:
        """Generate `n` completions for a single chat conversation.

        Returns a list of length n (each a GenerationResult with logprobs).
        """
        assert self._llm is not None, "Call .load() first."
        prompt = self._render(messages)
        params = self._sampling_params(temperature, max_tokens, n, stop,
                                       seed if seed is not None else self.seed)
        outputs = self._llm.generate([prompt], params, use_tqdm=False)
        return [self._to_result(c) for c in outputs[0].outputs]

    def chat_batch(self, batch_messages: List[List[Dict[str, str]]],
                   temperature: float = 0.0, max_tokens: int = 512,
                   stop: Optional[List[str]] = None,
                   seed: Optional[int] = None,
                   progress: bool = False) -> List[GenerationResult]:
        """Generate 1 completion for each of many conversations (batched).

        Returns a list aligned with `batch_messages`. This is the workhorse for
        batched execution — vLLM runs all prompts concurrently.
        """
        assert self._llm is not None, "Call .load() first."
        if not batch_messages:
            return []
        prompts = [self._render(m) for m in batch_messages]
        params = self._sampling_params(temperature, max_tokens, n=1, stop=stop,
                                       seed=seed if seed is not None else self.seed)
        outputs = self._llm.generate(prompts, params, use_tqdm=progress)
        return [self._to_result(o.outputs[0]) for o in outputs]

    def chat_batch_n(self, batch_messages: List[List[Dict[str, str]]], n: int,
                     temperature: float = 0.7, max_tokens: int = 512,
                     stop: Optional[List[str]] = None,
                     seed: Optional[int] = None,
                     progress: bool = False) -> List[List[GenerationResult]]:
        """Generate `n` completions for EACH of many conversations, in one batch.

        Used for self-consistency: all samples for all steps in a single GPU call.
        Returns list (aligned with batch_messages) of lists of length n.
        """
        assert self._llm is not None, "Call .load() first."
        if not batch_messages:
            return []
        prompts = [self._render(m) for m in batch_messages]
        params = self._sampling_params(temperature, max_tokens, n=n, stop=stop,
                                       seed=seed if seed is not None else self.seed)
        outputs = self._llm.generate(prompts, params, use_tqdm=progress)
        return [[self._to_result(c) for c in o.outputs] for o in outputs]

    def unload(self) -> None:
        """Release GPU memory (useful before switching agent<->judge model)."""
        import gc
        self._llm = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
