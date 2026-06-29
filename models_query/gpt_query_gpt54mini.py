from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Literal

import numpy as np
from openai import OpenAI

# ============================================================
# GPT-5.4 mini wrapper for the LLM-BLM project
# ============================================================
# If the API model id is different in your OpenAI account, change
# OPENAI_GPT_MODEL in your environment or edit DEFAULT_MODEL below.
# Example PowerShell:
#   $env:OPENAI_GPT_MODEL = "gpt-5.4-mini"
#   $env:OPENAI_API_KEY = "sk-..."

DEFAULT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-5.4-mini")


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
# Helpers
# -------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)

_EXPECTED_RETURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expected_return": {
            "type": "number",
            "description": "Expected monthly return as a decimal, for example 0.012 for 1.2%.",
        }
    },
    "required": ["expected_return"],
    "additionalProperties": False,
}


def _extract_json_object(text: str | None) -> dict[str, Any]:
    """
    Extract a JSON object from a model response.

    Accepts:
      - pure JSON: {"expected_return": 0.01}
      - fenced JSON: ```json {...} ```
      - JSON embedded in other text, best effort
    """
    if text is None:
        raise ValueError("Empty model response (None).")

    t = text.strip()
    if not t:
        raise ValueError("Empty model response ('').")

    m = _JSON_FENCE_RE.search(t)
    if m:
        return json.loads(m.group(1))

    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

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

    # Heuristic normalization: if the model returns percent, e.g. 2 for 2%, convert to decimal.
    if abs(xf) > 1.0 and abs(xf) <= 100.0:
        xf = xf / 100.0

    # Guardrail for monthly returns. This is intentionally wide.
    if abs(xf) > 1.0:
        raise ValueError(f"'expected_return' out of plausible monthly range: {xf}")

    return xf


def _responses_output_text(resp: Any) -> str:
    """
    Return text from an OpenAI Responses API object.
    The SDK usually exposes resp.output_text, but this fallback keeps the
    wrapper robust to small SDK response-shape differences.
    """
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))

    if chunks:
        return "\n".join(chunks)

    return str(resp)


# -------------------------
# Main GPT client wrapper
# -------------------------
class GPTQuery:
    """
    Wrapper around OpenAI for the LLM-BLM project.

    Main changes vs the older GPT wrapper:
      - defaults to GPT-5.4 mini through DEFAULT_MODEL;
      - uses the Responses API by default;
      - keeps a Chat Completions fallback for compatibility;
      - keeps the same sample_expected_return interface used by the project.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 5,
        retry_backoff_s: float = 1.0,
        timeout_s: Optional[float] = None,
        api_mode: Literal["responses", "chat"] = "responses",
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Pass api_key=... or set the OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=api_key, timeout=timeout_s) if timeout_s else OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.api_mode = api_mode

    def _call_responses_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """Call OpenAI Responses API and return raw text."""
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "expected_return_payload",
                        "schema": _EXPECTED_RETURN_SCHEMA,
                        "strict": True,
                    }
                },
            )
        except TypeError:
            # Older SDKs or model-specific endpoints may reject structured text arguments.
            # The prompt/parsing guardrails below still enforce JSON after the call.
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

        return _responses_output_text(resp)

    def _call_chat_completions_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """Compatibility fallback using Chat Completions."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except TypeError:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

        return resp.choices[0].message.content or ""

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

        # Strengthen the output contract without changing the calling code.
        json_instruction = (
            "\n\nReturn only valid JSON with exactly this schema: "
            '{"expected_return": <number>}. '
            "The value must be the expected monthly return in decimal form."
        )
        user_prompt_with_contract = user_prompt.rstrip() + json_instruction

        for _ in range(n_samples):
            ok = False
            last_err: Optional[str] = None

            for attempt in range(self.max_retries + 1):
                try:
                    if self.api_mode == "responses":
                        content = self._call_responses_api(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt_with_contract,
                            temperature=temperature,
                        )
                    else:
                        content = self._call_chat_completions_api(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt_with_contract,
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


# Optional alias if you want explicit naming elsewhere.
GPT54MiniQuery = GPTQuery


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Small smoke test for the GPT-5.4 mini expected-return wrapper.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_mode", type=str, choices=["responses", "chat"], default="responses")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    q = GPTQuery(model=args.model, api_mode=args.api_mode)
    result = q.sample_expected_return(
        system_prompt=(
            "You are a financial analyst. Estimate the next monthly stock return. "
            "Return only JSON."
        ),
        user_prompt="Estimate a plausible expected monthly return for a large-cap US equity.",
        n_samples=args.n_samples,
        temperature=args.temperature,
    )

    print(json.dumps(result.__dict__, indent=2))
