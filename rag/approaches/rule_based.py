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
- A rule message from a grammar checker (hint).
- A user sentence that contains the mistake.

Steps:
1. Identify the mistake using the rule message and the sentence.
2. Choose a clear grammar topic name.
3. Write a concise lesson (plain language).
4. Write one short exercise (similar style to the user's sentence).
5. If the rule message is vague, infer the topic mainly from the sentence.

Respond with JSON only (no markdown, no preamble). Use exactly these keys: topic, lesson, exercise.

Examples of the required shape (follow closely):

---
Rule message: Did you mean "it's" (it is)?
User sentence: Its a sunny day today.
JSON:
{"topic": "It's vs its", "lesson": "Use it's (with apostrophe) for it is or it has. Use its (no apostrophe) for possession, like his or hers.", "exercise": "Fill in: _____ (Its/It's) going to rain later."}
---
Rule message: Use third-person singular verb with he/she/it.
User sentence: He walk to school every day.
JSON:
{"topic": "Subject–verb agreement (he/she/it)", "lesson": "With he, she, or it, the present simple verb usually takes -s: he walks, she runs, it works.", "exercise": "Correct this: She study history on Tuesdays."}
---
"""


def _user_message(rule_message: str, example_sentence: str) -> str:
    return (
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
    """LLM path uses chat API + optional JSON schema; rule path uses LT strings only."""

    def __init__(self, llm: Optional["BaseLLM"] = None):
        self.llm = llm
        self._last_llm_result: dict = {}

    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        if self.llm is not None:
            return self._build_explanation_llm(context, topic)
        return self._build_explanation_rule_based(context, topic)

    def _build_explanation_llm(self, context: ContextAssembly, topic: str) -> str:
        self._last_llm_result = {}
        rule_message = ""
        example_sentence = ""
        if context.detected_mistake_examples:
            p = context.detected_mistake_examples[0]
            rule_message = p.rule_message or ""
            examples = p.examples
            example_sentence = (examples[0] if examples else "") or ""

        messages = [
            {"role": "system", "content": _SYSTEM_LESSON_INSTRUCTIONS},
            {"role": "user", "content": _user_message(rule_message, example_sentence)},
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

    def _build_explanation_rule_based(self, context: ContextAssembly, topic: str) -> str:
        self._last_llm_result = {}
        parts: List[str] = []
        if context.detected_mistake_examples:
            p = context.detected_mistake_examples[0]
            mistake_type_desc = p.description
            rule_message = p.rule_message
            if rule_message:
                parts.append(f"LanguageTool: {rule_message}")
            elif mistake_type_desc:
                parts.append(f"Mistake Type: {mistake_type_desc}")
        if context.long_term_dynamics:
            summary_content = context.long_term_dynamics[0].get("content", "")
            if summary_content:
                parts.append(f"Context: {summary_content[:200]}")
        if not parts:
            parts.append(f"Topic: {topic}")
        return " ".join(parts)[:500]

    def generate_exercises(self, primary_mistake_context: Optional[Any]) -> List[str]:
        if self.llm and self._last_llm_result:
            return self._last_llm_result.get("exercises", [])
        exercises: List[str] = []
        if primary_mistake_context:
            examples = (
                primary_mistake_context.examples
                if hasattr(primary_mistake_context, "examples")
                else (
                    primary_mistake_context.get("examples", [])
                    if isinstance(primary_mistake_context, dict)
                    else []
                )
            )
            if examples:
                for example in examples[:2]:
                    exercises.append(f"Practice: {example}")
        if not exercises:
            exercises.append("Complete the sentence with the correct form.")
            exercises.append("Identify and correct the mistake in the given text.")
        return exercises[:3]
