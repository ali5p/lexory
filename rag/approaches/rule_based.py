from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Mapping, Optional

import requests

from core.models import ContextAssembly
from .base import BaseApproach

if TYPE_CHECKING:
    from llm.base import BaseLLM

_log = logging.getLogger(__name__)

# Ollama structured output: JSON Schema for lesson payload (keys match historical prompt).
LESSON_RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "Short grammar topic title."},
        "lesson": {
            "type": "string",
            "description": "Brief explanation suitable for a learner.",
        },
    },
    "required": ["topic", "lesson"],
}

_SYSTEM_LESSON_INSTRUCTIONS = """You are an English teacher who teaches deductively: state the grammar rule first, then show a short example. Produce ONE short grammar lesson as structured JSON only.

The user message will contain:
- A mistake category (Lexory taxonomy label).
- A rule message from a grammar checker (hint).
- A user sentence that contains the mistake.

Steps:
1. The rule message identifies ONE specific error — the sentence may contain other errors; ignore them.
2. Teach ONLY the error described in the rule message (not other mistakes in the same sentence).
3. Choose a clear grammar topic name matching that specific error.
4. In the lesson, state the rule in plain language FIRST, then give one short correct example.
5. If the rule message is vague, infer the rule from the rule message first, not from unrelated words in the sentence.

Respond with JSON only (no markdown, no preamble). Use exactly these keys: topic, lesson.

Examples of the required shape (follow closely):

---
Mistake category: articles
Rule message: Did you mean "it's" (it is)?
User sentence: Its a sunny day today.
JSON:
{"topic": "It's vs its", "lesson": "Use it's (with apostrophe) for it is or it has. Use its (no apostrophe) for possession, like his or hers."}
---
Mistake category: subject_verb_agreement
Rule message: Use third-person singular verb with he/she/it.
User sentence: He walk to school every day.
JSON:
{"topic": "Subject–verb agreement (he/she/it)", "lesson": "With he, she, or it, the present simple verb usually takes -s: he walks, she runs, it works."}
---
"""


def _user_message(
    rule_message: str,
    example_sentence: str,
    mistake_type: str = "",
    rule_id: str = "",
) -> str:
    category = mistake_type.strip() or "(none)"
    focus = rule_id.strip() or "(none)"
    return (
        f"Mistake category:\n{category}\n\n"
        f"Focus rule id (ONLY this error — ignore others in the sentence):\n{focus}\n\n"
        f"Rule message from grammar checker:\n{rule_message or '(none)'}\n\n"
        f"User sentence (may contain other errors — do not teach those):\n{example_sentence or '(none)'}\n"
    )


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON object from text (handles markdown code blocks or extra text)."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_lesson_json(response_text: str) -> Optional[dict]:
    text = response_text.strip()
    for candidate in (text, _extract_json(response_text) or ""):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


class RuleBasedApproach(BaseApproach):
    """Deductive lesson generation via the LLM chat API (JSON-schema constrained):
    state the rule first, then a short example. On any failure, returns an error
    lesson (``generation_status='error'``) so the API response stays valid.

    Subclasses change the teaching style by overriding ``SYSTEM_PROMPT`` and the
    user message; the LLM call, parsing, and error handling are shared.
    """

    SYSTEM_PROMPT = _SYSTEM_LESSON_INSTRUCTIONS

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm
        self._last_llm_result: dict = {}

    @staticmethod
    def _primary_fields(context: ContextAssembly) -> tuple[str, str, str, str]:
        """(rule_message, example_sentence, mistake_type, rule_id) of the primary mistake."""
        rule_message = example_sentence = mistake_type = rule_id = ""
        if context.detected_mistake_examples:
            p = context.detected_mistake_examples[0]
            rule_message = p.rule_message or ""
            mistake_type = p.mistake_type or ""
            rule_id = p.rule_id or ""
            examples = p.examples
            example_sentence = (examples[0] if examples else "") or ""
        return rule_message, example_sentence, mistake_type, rule_id

    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        rule_message, example_sentence, mistake_type, rule_id = self._primary_fields(context)
        user_message = _user_message(rule_message, example_sentence, mistake_type, rule_id)
        return self._generate(self.SYSTEM_PROMPT, user_message, topic)

    def _generate(self, system_prompt: str, user_message: str, topic: str) -> str:
        """Shared LLM call + JSON parse + error handling. Populates _last_llm_result."""
        self._last_llm_result = {}
        messages: list[Mapping[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response_text = self.llm.chat(
                messages,
                temperature=0.0,
                json_schema=LESSON_RESPONSE_JSON_SCHEMA,
            )
        except requests.RequestException as e:
            return self._fail(f"LLM generation failed: {e!s}")
        except (KeyError, ValueError, TypeError) as e:
            return self._fail(f"LLM generation failed: invalid response ({e!s})")
        except Exception as e:
            _log.exception("LLM generation failed")
            return self._fail(f"LLM generation failed: {e!s}")

        parsed = _parse_lesson_json(response_text)
        if not parsed:
            return self._fail("LLM generation failed: invalid JSON response")

        required_keys = ("topic", "lesson")
        if not all(k in parsed for k in required_keys):
            missing = [k for k in required_keys if k not in parsed]
            return self._fail(f"LLM generation failed: missing keys {missing}")

        self._last_llm_result = {
            "topic": str(parsed.get("topic", "")).strip() or topic,
            "explanation": str(parsed.get("lesson", "")).strip(),
            "generation_status": "ok",
        }
        return self._last_llm_result["explanation"]

    def _fail(self, err_msg: str) -> str:
        self._last_llm_result = {
            "topic": "error",
            "explanation": err_msg,
            "generation_status": "error",
        }
        return err_msg
