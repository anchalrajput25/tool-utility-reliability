"""Free judge: the paper's BFCL judge prompt, served by any OpenAI-compatible
chat endpoint (default: Google Gemini free tier).

Prompt and decision rule are copied verbatim from the paper's scorer
(ToCall_or_NotToCall/src/scorer.py, LLMJudgeScorer). Only the model and API
differ from the paper (GPT-4o via the Responses API) -- report this.

Environment:
  JUDGE_API_KEY   key for the provider
  JUDGE_BASE_URL  default https://generativelanguage.googleapis.com/v1beta/openai/
  JUDGE_MODEL     default gemini-2.5-flash
  JUDGE_RPM       requests per minute to stay under the free limit (default 10)
"""
from __future__ import annotations
import json, os, re, time

_JUDGE_SYSTEM = (
    "You are a strict grading engine for short-answer trivia questions.\n"
    "Given a QUESTION, a REFERENCE (correct) answer, and a model's RESPONSE, "
    "judge whether the RESPONSE is factually equivalent to the REFERENCE.\n"
    "- Ignore differences in phrasing, formatting, or extra context.\n"
    "- Set is_correct=true only if the core factual content matches the reference.\n"
    "- Set is_correct=false if the response is wrong, missing, or contradicts the reference.\n"
    "Return ONLY valid JSON matching the provided schema."
)
_SCHEMA_HINT = '\nSchema: {"is_correct": boolean, "reason": string}'


class FreeJudge:
    def __init__(self, model: str | None = None, temperature: float = 1.0):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["JUDGE_API_KEY"],
            base_url=os.environ.get("JUDGE_BASE_URL",
                                    "https://generativelanguage.googleapis.com/v1beta/openai/"),
        )
        self.model = model or os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.min_gap = 60.0 / float(os.environ.get("JUDGE_RPM", "10"))
        self._last = 0.0

    def score(self, response: str, ground_truth, question: str, **_) -> float:
        if not str(response).strip() or ground_truth is None:
            return 0.0
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM + _SCHEMA_HINT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nREFERENCE:\n{ground_truth}\n\nRESPONSE:\n{response}"},
        ]
        for attempt in range(6):
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last = time.time()
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature)
                return _parse(r.choices[0].message.content)
            except Exception as e:  # rate limits / transient errors
                print(f"[judge retry {attempt + 1}] {str(e)[:120]}")
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError("judge failed after retries")


def _parse(text: str | None) -> float:
    if not text:
        return 0.0
    m = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, AttributeError):
        return 0.0
    return 1.0 if verdict.get("is_correct") is True else 0.0
