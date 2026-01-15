# gpt_query.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "Le package 'openai' n'est pas installé. Fais: pip install openai"
    ) from e


# -------------------------
# Public result container
# -------------------------
@dataclass
class LLMReturnResult:
    samples: list[float]
    n_success: int
    mean: Optional[float]
    errors: list[str]


# -------------------------
# Helpers (GPT-specific I/O)
# -------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a model response.
    Accepts:
      - pure JSON: {"expected_return": 0.01}
      - fenced JSON: ```json {...} ```
      - JSON embedded in other text (best effort)
    """
    if text is None:
        raise ValueError("Empty model response (None).")

    t = text.strip()

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
    """
    Validate that payload contains {"expected_return": number}.
    Returns expected_return as float.
    """
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
# Main GPT client wrapper
# -------------------------
class GPTQuery:
    """
    Small wrapper around OpenAI Chat Completions for the project.

    Goal:
      - isolate OpenAI-specific code here
      - return clean numeric samples and diagnostics
      - later you can create llama_query.py etc. with same method signature
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 5,
        retry_backoff_s: float = 1.0,
        timeout_s: Optional[float] = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout_s) if timeout_s else OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)

    def sample_expected_return(
        self,
        system_prompt: str,
        user_prompt: str,
        n_samples: int = 5,
        temperature: float = 0.5,
    ) -> LLMReturnResult:
        """
        Query GPT n_samples times and parse {"expected_return": number}.
        """
        samples: list[float] = []
        errors: list[str] = []

        n_samples = int(n_samples)
        temperature = float(temperature)

        for _ in range(n_samples):
            ok = False
            last_err: Optional[str] = None

            for attempt in range(self.max_retries + 1):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                    )

                    content = resp.choices[0].message.content
                    payload = _extract_json_object(content)
                    er = _validate_expected_return(payload)

                    samples.append(float(er))
                    ok = True
                    break

                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    # retry with exponential backoff
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
