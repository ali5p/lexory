#!/usr/bin/env python3
"""
Extract LanguageTool rule IDs for English (en-US) and split them into
grammar vs style assets.

Build-time only. Deterministic. No runtime LanguageTool usage.

Outputs:
- assets/languagetool_rule_ids_en_grammar.json
- assets/languagetool_rule_ids_en_style.json
- assets/languagetool_rule_ids_en_all.json
"""

from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET


LT_VERSION = "6.6"


def extract_rule_ids(xml_path: Path) -> set[str]:
    rule_ids: set[str] = set()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for rule in root.iter("rule"):
        rule_id = rule.attrib.get("id")
        if rule_id:
            rule_ids.add(rule_id)

    return rule_ids


def main() -> None:
    project_root = Path(__file__).parent.parent
    lt_root = project_root / "vendor" / "languagetool" / LT_VERSION / "rules"

    grammar_ids: set[str] = set()
    style_ids: set[str] = set()

    # Core grammar rules
    grammar_xml = lt_root / "en" / "grammar.xml"
    if not grammar_xml.exists():
        print(f"Missing grammar.xml: {grammar_xml}", file=sys.stderr)
        sys.exit(1)

    grammar_ids.update(extract_rule_ids(grammar_xml))

    # Core style rules
    style_xml = lt_root / "en" / "style.xml"
    if style_xml.exists():
        style_ids.update(extract_rule_ids(style_xml))

    # en-US overrides (treated as style)
    en_us_dir = lt_root / "en" / "en-US"
    if en_us_dir.exists():
        for xml_file in sorted(en_us_dir.glob("*.xml")):
            style_ids.update(extract_rule_ids(xml_file))

    all_ids = grammar_ids | style_ids

    assets_dir = project_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, data: set[str]) -> None:
        path = assets_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(sorted(data), f, indent=2, ensure_ascii=False)

    dump("languagetool_rule_ids_en_grammar.json", grammar_ids)
    dump("languagetool_rule_ids_en_style.json", style_ids)
    dump("languagetool_rule_ids_en_all.json", all_ids)

    print("Language: en-US")
    print(f"LanguageTool version: {LT_VERSION}")
    print(f"Grammar rule IDs: {len(grammar_ids)}")
    print(f"Style rule IDs: {len(style_ids)}")
    print(f"Total rule IDs: {len(all_ids)}")
    print("Assets written to:", assets_dir)


if __name__ == "__main__":
    main()
