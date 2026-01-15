from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import ollama  # type: ignore
except ImportError as e:
    raise ImportError("Le package 'ollama' n'est pas installé. Fais: pip install ollama") from e


# -------------------------
# Public result container (same contract as gpt_query.py)
# -------------------------
@dataclass
class LLMReturnResult:
    samples: list[float]
    n_success: int
    mean: Optional[float]
    errors: list[str]


# -------------------------
# Helpers (JSON extraction/validation)
# -------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a model response.

    Accepts:
      - pure JSON: {"expected_return": 0.01}
      - fenced JSON: ```json {...} ```
      - JSON embedded in other text (best effort)
    """
    if text is None:
        raise ValueError("Empty model response (None).")

    t = str(text).strip()

    # 1) fenced JSON
    m = _JSON_FENCE_RE.search(t)
    if m:
        return json.loads(m.group(1))

    # 2) direct JSON
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    # 3) fallback: first {...} chunk
    m = _FIRST_JSON_OBJ_RE.search(t)
    if m:
        return json.loads(m.group(1))

    raise ValueError(f"Could not extract JSON from response: {t[:200]!r}")


def _validate_expected_return(payload: dict[str, Any]) -> float:
    """Validate that payload contains {"expected_return": number}."""
    if "expected_return" not in payload:
        raise ValueError(f"Missing key 'expected_return' in payload keys={list(payload.keys())}")

    x = payload["expected_return"]
    try:
        xf = float(x)
    except Exception as e:
        raise ValueError(f"'expected_return' not convertible to float: {x!r}") from e

    if not np.isfinite(xf):
        raise ValueError(f"'expected_return' is not finite: {xf}")

    return xf


# -------------------------
# Main Ollama client wrapper
# -------------------------
class GemmaQuery:
    """Small wrapper around Ollama Chat for this project.

    Goal:
      - isolate Ollama-specific code here
      - return clean numeric samples and diagnostics
      - keep the same method signature as GPTQuery

    Notes:
      - You can use ANY Ollama model id (e.g. "gemma3", "llama3.1", etc.)
      - `host` should usually be "http://localhost:11434"
    """

    def __init__(
        self,
        model: str = "gemma3",
        host: str = "http://localhost:11434",
        max_retries: int = 5,
        retry_backoff_s: float = 1.0,
        timeout_s: Optional[float] = None,
        force_json: bool = False,
    ) -> None:
        self.model = str(model)
        self.host = str(host)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.timeout_s = timeout_s
        self.force_json = bool(force_json)

        # Prefer Client if available (lets us set host)
        self._client = None
        Client = getattr(ollama, "Client", None)
        if Client is not None:
            try:
                # Some versions accept timeout as a kwarg; keep optional.
                if timeout_s is not None:
                    self._client = Client(host=self.host, timeout=timeout_s)
                else:
                    self._client = Client(host=self.host)
            except TypeError:
                # Fallback if timeout kwarg not supported
                self._client = Client(host=self.host)

    def _chat_once(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Single Ollama chat call. Returns raw text content."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        options: dict[str, Any] = {
            "temperature": float(temperature),
        }

        # Some Ollama versions support `format="json"` to enforce JSON.
        # We keep it optional because not all versions/models support it.
        kwargs: dict[str, Any] = {}
        if self.force_json:
            kwargs["format"] = "json"

        if self._client is not None:
            try:
                resp = self._client.chat(model=self.model, messages=messages, options=options, **kwargs)
            except TypeError:
                # If `format` isn't accepted, retry without it
                kwargs.pop("format", None)
                resp = self._client.chat(model=self.model, messages=messages, options=options)
        else:
            # Module-level fallback (host driven by OLLAMA_HOST env var)
            try:
                resp = ollama.chat(model=self.model, messages=messages, options=options, **kwargs)  # type: ignore
            except TypeError:
                kwargs.pop("format", None)
                resp = ollama.chat(model=self.model, messages=messages, options=options)  # type: ignore

        # Response shape usually: {"message": {"role": "assistant", "content": "..."}, ...}
        msg = (resp or {}).get("message") or {}
        content = msg.get("content")
        if content is None:
            # Best-effort fallback: some builds return {"response": "..."}
            content = (resp or {}).get("response")

        if content is None:
            raise ValueError(f"Empty response content from Ollama (keys={list((resp or {}).keys())}).")

        return str(content)

    def sample_expected_return(
        self,
        system_prompt: str,
        user_prompt: str,
        n_samples: int = 5,
        temperature: float = 0.5,
    ) -> LLMReturnResult:
        """Query Ollama n_samples times and parse {"expected_return": number}."""
        samples: list[float] = []
        errors: list[str] = []

        n_samples = int(n_samples)
        temperature = float(temperature)

        for _ in range(n_samples):
            ok = False
            last_err: Optional[str] = None

            for attempt in range(self.max_retries + 1):
                try:
                    content = self._chat_once(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                    )

                    payload = _extract_json_object(content)
                    er = _validate_expected_return(payload)

                    samples.append(float(er))
                    ok = True
                    break

                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_s * (2 ** attempt))
                    else:
                        break

            if not ok and last_err:
                errors.append(last_err)

        mean = float(np.mean(samples)) if samples else None
        return LLMReturnResult(
            samples=samples,
            n_success=len(samples),
            mean=mean,
            errors=errors,
        )
