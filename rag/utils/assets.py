"""Asset loading utilities for LanguageTool pipeline."""
import json
from pathlib import Path
from typing import Dict

ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"


def load_languagetool_mapping() -> Dict[str, str]:
    """Load LanguageTool ruleId to mistake_type mapping."""
    mapping_path = ASSETS_DIR / "languagetool_to_mistaketype.json"
    if not mapping_path.exists():
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mistake_logic_vocab() -> Dict[str, int]:
    """Load mistake_type vocabulary and return mapping to index."""
    vocab_path = ASSETS_DIR / "mistake_logic_vocab.json"
    if not vocab_path.exists():
        return {"other": 0}
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("categories", ["other"])
