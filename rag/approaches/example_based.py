from __future__ import annotations

from typing import Mapping, Sequence

from core.models import ContextAssembly

from .rule_based import RuleBasedApproach

_SYSTEM_EXAMPLE_BASED_INSTRUCTIONS = """You are an English teacher who teaches by example (inductively). Produce ONE short grammar lesson as structured JSON only.

The user message will contain:
- A mistake category (Lexory taxonomy label).
- A rule message from a grammar checker (hint).
- The learner's CURRENT sentence with the mistake.
- A few of the learner's OWN past sentences with the same kind of mistake.

Teach inductively:
1. Point out the pattern shared by the learner's sentences.
2. For each wrong sentence, show its corrected version so the contrast is visible.
3. Keep rule jargon minimal; let the corrected examples make the rule obvious.

Respond with JSON only (no markdown, no preamble). Use exactly these keys: topic, lesson.

Example of the required shape (follow closely):
---
Mistake category: subject_verb_agreement
Past sentences:
- He walk to school every day.
- She play tennis on weekends.
Current sentence: My brother live in Berlin.
JSON:
{"topic": "Subject-verb agreement (he/she/it)", "lesson": "Across your sentences the verb needs -s after a singular subject: 'He walk' becomes 'He walks', 'She play' becomes 'She plays', 'My brother live' becomes 'My brother lives'. With he, she, it, or one person, add -s to the present-simple verb."}
---
"""


def _example_based_user_message(
    mistake_type: str,
    current_sentence: str,
    rule_message: str,
    rule_id: str,
    past_examples: Sequence[Mapping[str, str]],
) -> str:
    category = mistake_type.strip() or "(none)"
    parts = [
        f"Mistake category:\n{category}\n",
        f"Focus rule id (ONLY this error — ignore others in the sentence):\n"
        f"{rule_id.strip() or '(none)'}\n",
        f"Rule message from grammar checker:\n{rule_message or '(none)'}\n",
        f"Current sentence (may contain other errors — do not teach those):\n"
        f"{current_sentence or '(none)'}\n",
    ]
    texts = [str(e.get("text", "")).strip() for e in past_examples]
    texts = [t for t in texts if t]
    if texts:
        joined = "\n".join(f"- {t}" for t in texts)
        parts.append(f"The learner's own past sentences with this mistake:\n{joined}\n")
    return "\n".join(parts)


class ExampleBasedApproach(RuleBasedApproach):
    """Inductive teaching: build the lesson from the learner's own prior sentences
    (``ContextAssembly.similar_past_examples``) as visible contrast pairs. Reuses
    the shared LLM call / parsing / error handling from ``RuleBasedApproach``.
    """

    SYSTEM_PROMPT = _SYSTEM_EXAMPLE_BASED_INSTRUCTIONS

    def build_explanation(self, context: ContextAssembly, topic: str) -> str:
        rule_message, example_sentence, mistake_type, rule_id = self._primary_fields(context)
        user_message = _example_based_user_message(
            mistake_type=mistake_type,
            current_sentence=example_sentence,
            rule_message=rule_message,
            rule_id=rule_id,
            past_examples=context.similar_past_examples,
        )
        return self._generate(self.SYSTEM_PROMPT, user_message, topic)
