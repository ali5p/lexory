"""Normalize LanguageTool rule_id for stable taxonomy mapping and vector retrieval."""
import re
from typing import Union


def normalize_rule_id(raw: Union[str, None]) -> str:
    """
    Normalize rule_id from language_tool_python Match for stable taxonomy lookup.

    LanguageTool can return:
    - Base form: "BASE_FORM", "EN_A_VS_AN"
    - Sub-rule form: "TOT_HE[1]", "MORFOLOGIK_RULE_EN_US[42]"

    Our mapping (languagetool_to_mistaketype.json) uses base IDs without suffixes.
    Downstream (taxonomy mapping, mistake_logic_vector, Qdrant payloads) expect
    a consistent uppercase string.

    Returns:
        Uppercase base rule ID, stripped of [N] suffix. Empty string if input is falsy.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Strip sub-rule suffix: "TOT_HE[1]" -> "TOT_HE"
    base = re.sub(r"\[\d+\]$", "", s)
    return base.upper() if base else ""
