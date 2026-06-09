from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, List, Optional

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
        "exercise": {
            "type": "string",
            "description": "One short practice prompt similar to the user's mistake.",
        },
    },
    "required": ["topic", "lesson", "exercise"],
}

_SYSTEM_LESSON_INSTRUCTIONS = """You are an English teacher. Your task is to produce ONE short grammar lesson as structured JSON only.

The user message will contain:
- A mistake category (Lexory taxonomy label).
- A rule message from a grammar checker (hint).
- A user sentence that contains the mistake.

Steps:
1. Identify the mistake using the category, rule message, and the sentence.
2. Choose a clear grammar topic name.
3. Write a concise lesson (plain language).
4. Write one short exercise (similar style to the user's sentence).
5. If the rule message is vague, infer the topic mainly from the sentence.

Respond with JSON only (no markdown, no preamble). Use exactly these keys: topic, lesson, exercise.

Examples of the required shape (follow closely):

---
Mistake category: articles
Rule message: Did you mean "it's" (it is)?
User sentence: Its a sunny day today.
JSON:
{"topic": "It's vs its", "lesson": "Use it's (with apostrophe) for it is or it has. Use its (no apostrophe) for possession, like his or hers.", "exercise": "Fill in: _____ (Its/It's) going to rain later."}
---
Mistake category: subject_verb_agreement
Rule message: Use third-person singular verb with he/she/it.
User sentence: He walk to school every day.
JSON:
{"topic": "Subject–verb agreement (he/she/it)", "lesson": "With he, she, or it, the present simple verb usually takes -s: he walks, she runs, it works.", "exercise": "Correct this: She study history on Tuesdays."}
---
"""


def _user_message(
    rule_message: str,
    example_sentence: str,
    mistake_type: str = "",
) -> str:
    category = mistake_type.strip() or "(none)"
    return (
        f"Mistake category:\n{category}\n\n"
        f"Rule message from grammar checker:\n{rule_message or '(none)'}\n\n"
        f"User sentence with the mistake:\n{example_sentence or '(none)'}\n"
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
    """Generate a lesson via the LLM chat API (JSON-schema constrained). On any
    failure, return an error lesson (approach_type='llm_error') so the API
    response stays structurally valid."""

    def __init__(self, llm: "BaseLLM"):
        self.llm = llm
        self._last_llm_result: dict = {}

    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        self._last_llm_result = {}
        rule_message = ""
        example_sentence = ""
        mistake_type = ""
        if context.detected_mistake_examples:
            p = context.detected_mistake_examples[0]
            rule_message = p.rule_message or ""
            mistake_type = p.mistake_type or ""
            examples = p.examples
            example_sentence = (examples[0] if examples else "") or ""

        messages = [
            {"role": "system", "content": _SYSTEM_LESSON_INSTRUCTIONS},
            {
                "role": "user",
                "content": _user_message(rule_message, example_sentence, mistake_type),
            },
        ]

        try:
            response_text = self.llm.chat(
                messages,
                temperature=0.0,
                json_schema=LESSON_RESPONSE_JSON_SCHEMA,
            )
        except requests.RequestException as e:
            err_msg = f"LLM generation failed: {e!s}"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg
        except (KeyError, ValueError, TypeError) as e:
            err_msg = f"LLM generation failed: invalid response ({e!s})"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg
        except Exception as e:
            _log.exception("LLM generation failed")
            err_msg = f"LLM generation failed: {e!s}"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg

        parsed = _parse_lesson_json(response_text)
        if not parsed:
            err_msg = "LLM generation failed: invalid JSON response"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg

        required_keys = ("topic", "lesson", "exercise")
        if not all(k in parsed for k in required_keys):
            missing = [k for k in required_keys if k not in parsed]
            err_msg = f"LLM generation failed: missing keys {missing}"
            self._last_llm_result = {
                "topic": "error",
                "explanation": err_msg,
                "exercises": [],
                "approach_type": "llm_error",
            }
            return err_msg

        self._last_llm_result = {
            "topic": str(parsed.get("topic", "")).strip() or topic,
            "explanation": str(parsed.get("lesson", "")).strip(),
            "exercises": [str(parsed.get("exercise", "")).strip()]
            if parsed.get("exercise")
            else [],
            "approach_type": "llm",
        }
        return self._last_llm_result["explanation"]

    def generate_exercises(self, primary_mistake_context: Optional[Any] = None) -> List[str]:
        return self._last_llm_result.get("exercises", [])
