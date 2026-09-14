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
    """Persist the configured model exactly; never silently substitute one."""
    from ..utils.io import save_json, load_json
    path = os.path.join(cfg.path("data_processed"), "agent_model.json")
    base = cfg.models.agent
    if os.path.exists(path):
        cached = load_json(path)
        if cached.get("name") != base.name or cached.get("dtype") != base.dtype:
            raise ValueError("Cached agent model differs from configuration; use a new run directory")
        return cached

    vram = gpu_vram_gb()
    choice = {"name": base.name, "dtype": base.dtype,
              "gpu_memory_utilization": base.gpu_memory_utilization,
              "vram_gb": round(vram, 1) if vram else None,
              "reason": "exact configured model; no automatic model or quantization fallback"}
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
    prompt_tokens: Optional[int] = None

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "token_ids": self.token_ids,
                "tokens": [t.to_dict() for t in self.tokens],
                "prompt_tokens": self.prompt_tokens}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GenerationResult":
        return GenerationResult(
            text=d["text"], token_ids=d.get("token_ids", []),
            tokens=[TokenInfo.from_dict(t) for t in d.get("tokens", [])],
            prompt_tokens=d.get("prompt_tokens"),
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
    """Use the configured judge; a smaller judge requires explicit opt-in."""
    jc = cfg.raw["models"]["judge"]
    if jc["fallback"]["use_fallback"]:
        return {"name": jc["fallback"]["name"], "dtype": "auto",
                "gpu_memory_utilization": jc["gpu_memory_utilization"],
                "reason": "forced fallback via config"}
    return {"name": jc["name"], "dtype": jc["dtype"],
            "gpu_memory_utilization": jc["gpu_memory_utilization"],
            "reason": "exact configured judge; no automatic fallback"}


class VLLMClient:
    """Thin wrapper around a vLLM engine with logprob extraction."""

    def __init__(self, model_name: str, dtype: str = "float16",
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.90,
                 logprobs_topk: int = 20, trust_remote_code: bool = True,
                 seed: int = 0, enable_prefix_caching: Optional[bool] = None):
        self.model_name = model_name
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.logprobs_topk = logprobs_topk
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.enable_prefix_caching = enable_prefix_caching
        self._llm = None
        self._tokenizer = None

    # --------------------------------------------------------------------- #
    def load(self) -> "VLLMClient":
        """Instantiate the vLLM engine (lazy import). Call once per session."""
        from vllm import LLM
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code)
        cache_options = ({} if self.enable_prefix_caching is None else
                         {"enable_prefix_caching": self.enable_prefix_caching})
        self._llm = LLM(
            model=self.model_name,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
            seed=self.seed,
            **cache_options,
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

    def _to_result(self, comp, prompt_tokens=None) -> GenerationResult:
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
        return GenerationResult(text=comp.text, token_ids=list(comp.token_ids), tokens=tokens,
                                prompt_tokens=prompt_tokens)

    @staticmethod
    def _prompt_tokens(output):
        ids = getattr(output, "prompt_token_ids", None)
        return None if ids is None else len(ids)

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
        return [self._to_result(c, self._prompt_tokens(outputs[0])) for c in outputs[0].outputs]

    def chat_batch(self, batch_messages: List[List[Dict[str, str]]],
                   temperature: float = 0.0, max_tokens: int | List[int] = 512,
                   stop: Optional[List[str]] = None,
                   seed: Optional[int] = None,
                   progress: bool = False) -> List[GenerationResult]:
        """Generate 1 completion for each of many conversations (batched).

        Returns a list aligned with `batch_messages`. This is the workhorse for
        batched execution — vLLM runs all prompts concurrently.
        `max_tokens` may be a shared limit or one positive limit per prompt.
        """
        assert self._llm is not None, "Call .load() first."
        if not batch_messages:
            return []
        prompts = [self._render(m) for m in batch_messages]
        if isinstance(max_tokens, list):
            if len(max_tokens) != len(batch_messages):
                raise ValueError("Token limits must match the number of prompts")
            if any(not isinstance(limit, int) or limit <= 0 for limit in max_tokens):
                raise ValueError("Token limits must be positive integers")
            params = [self._sampling_params(
                temperature, limit, n=1, stop=stop,
                seed=seed if seed is not None else self.seed,
            ) for limit in max_tokens]
        else:
            params = self._sampling_params(temperature, max_tokens, n=1, stop=stop,
                                           seed=seed if seed is not None else self.seed)
        outputs = self._llm.generate(prompts, params, use_tqdm=progress)
        return [self._to_result(o.outputs[0], self._prompt_tokens(o)) for o in outputs]

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
        return [[self._to_result(c, self._prompt_tokens(o)) for c in o.outputs] for o in outputs]

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
