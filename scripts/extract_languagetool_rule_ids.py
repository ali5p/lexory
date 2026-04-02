#!/usr/bin/env python3
"""
Extract LanguageTool English rule IDs from the same XML tree LT ships (all `en`
resources), and optionally bulk-fill `languagetool_to_mistaketype.json`.

Build-time only. Deterministic. Does not call a running LanguageTool server.

Source layout (either works):
  1) Legacy mirror: vendor/languagetool/<version>/rules/en/
  2) Upstream repo: .../languagetool-language-modules/en/src/main/resources/org/languagetool/rules/en/

We scan every `*.xml` under that directory (grammar + style + regional variants).
Collects `id` on `<rule>` and `<rulegroup>` (namespace-aware), using iterparse for
large files.

Environment:
  LT_RULES_EN_ROOT   Absolute path to the `en` rules directory (overrides defaults)
  LT_GIT_REF         Optional tag/sha recorded in the manifest (e.g. v6.7)

Outputs (assets/):
  - languagetool_rule_ids_en_all.json
  - languagetool_rule_inventory_manifest.json  (when --write-manifest)

With --bulk-fill-mapping:
  - Merges any ID in _all_ that is missing from languagetool_to_mistaketype.json
    using --default-new (default: unmapped). Existing entries are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

# Default LT release to record in the manifest (align Docker image / public API if you can).
DEFAULT_LT_GIT_REF = os.environ.get("LT_GIT_REF", "v6.6")

DEFAULT_RULES_REL_LEGACY = Path("vendor") / "languagetool" / DEFAULT_LT_GIT_REF.lstrip("v") / "rules" / "en"
DEFAULT_RULES_REL_UPSTREAM = (
    Path("vendor")
    / "lt-en-rules"
    / "languagetool-language-modules"
    / "en"
    / "src"
    / "main"
    / "resources"
    / "org"
    / "languagetool"
    / "rules"
    / "en"
)


def local_tag(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _is_rule_xml(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".xml"):
        return False
    if name == "pom.xml":
        return False
    if "owasp" in name or "circleci" in name:
        return False
    return True


def extract_rule_ids_from_xml(path: Path) -> set[str]:
    """Parse one XML file; return rule / rulegroup id attributes."""
    ids: set[str] = set()
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            tag = local_tag(elem.tag)
            if tag in ("rule", "rulegroup"):
                rid = elem.attrib.get("id", "").strip()
                if rid:
                    ids.add(rid)
            elem.clear()
    except ET.ParseError as e:
        print(f"Warning: XML parse error in {path}: {e}", file=sys.stderr)
    return ids


def iter_rule_xml_files(en_root: Path) -> Iterable[Path]:
    if not en_root.is_dir():
        return
    for path in sorted(en_root.rglob("*.xml")):
        if _is_rule_xml(path):
            yield path


def classify_file(en_root: Path, xml_path: Path) -> str:
    """
    Split into grammar vs style asset buckets (heuristic, same spirit as older script).
    Regional `style.xml` → style; `grammar*.xml` → grammar.
    """
    try:
        rel = xml_path.relative_to(en_root)
    except ValueError:
        rel = xml_path
    parts = [p.lower() for p in rel.parts]
    name = rel.name.lower()

    if name == "style.xml" or "style" in name:
        return "style"
    # Old script treated en-US XMLs as style bucket — keep regional style-ish split.
    if "en-us" in parts and name != "grammar.xml":
        return "style"
    return "grammar"


def discover_rules_en_root(project_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_dir() else None
    env = os.environ.get("LT_RULES_EN_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    candidates = [
        project_root / DEFAULT_RULES_REL_UPSTREAM,
        project_root / DEFAULT_RULES_REL_LEGACY,
        project_root / "vendor" / "languagetool" / "6.6" / "rules" / "en",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules-root",
        metavar="DIR",
        default=None,
        help="Path to LanguageTool `rules/en` directory (see script docstring).",
    )
    parser.add_argument(
        "--project-root",
        metavar="DIR",
        default=None,
        help="Defaults to parent of scripts/.",
    )
    parser.add_argument(
        "--lt-ref",
        default=DEFAULT_LT_GIT_REF,
        help="Git ref / version string stored in manifest (default from LT_GIT_REF or v6.6).",
    )
    parser.add_argument(
        "--bulk-fill-mapping",
        action="store_true",
        help="Add missing rule IDs to assets/languagetool_to_mistaketype.json.",
    )
    parser.add_argument(
        "--default-new",
        default="unmapped",
        help='mistake_type for newly added keys (default: "unmapped").',
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        default=True,
        help="Write languagetool_rule_inventory_manifest.json (default: on).",
    )
    parser.add_argument(
        "--no-write-manifest",
        action="store_false",
        dest="write_manifest",
        help="Skip manifest JSON.",
    )
    args = parser.parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parent.parent
    )
    en_root = discover_rules_en_root(project_root, args.rules_root)
    if en_root is None:
        print(
            "Could not find LanguageTool English rules directory.\n"
            "Set LT_RULES_EN_ROOT or pass --rules-root.\n"
            "Example (sparse clone):\n"
            "  git clone --depth 1 --branch v6.7 --filter=blob:none --sparse "
            "https://github.com/languagetool-org/languagetool.git vendor/lt-en-rules\n"
            "  cd vendor/lt-en-rules && git sparse-checkout set "
            "languagetool-language-modules/en/src/main/resources/org/languagetool/rules/en\n",
            file=sys.stderr,
        )
        sys.exit(1)

    grammar_ids: set[str] = set()
    style_ids: set[str] = set()
    file_counts: dict[str, int] = {}

    for xml_path in iter_rule_xml_files(en_root):
        ids = extract_rule_ids_from_xml(xml_path)
        if not ids:
            continue
        key = str(xml_path.relative_to(en_root))
        file_counts[key] = len(ids)
        bucket = classify_file(en_root, xml_path)
        if bucket == "style":
            style_ids.update(ids)
        else:
            grammar_ids.update(ids)

    all_ids = grammar_ids | style_ids
    assets_dir = project_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def dump_list(name: str, data: set[str]) -> None:
        path = assets_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(sorted(data), f, indent=2, ensure_ascii=False)
            f.write("\n")

    dump_list("languagetool_rule_ids_en_all.json", all_ids)

    if args.write_manifest:
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lt_git_ref": args.lt_ref,
            "rules_en_root": str(en_root),
            "scope_note": (
                "IDs come from rule/rulegroup id attributes in all *.xml under this English "
                "rules tree (grammar, style, en-US/en-GB/…). LT may report additional ids from "
                "spelling, n-gram, or server-only rules not defined in these XMLs."
            ),
            "xml_files_scanned": len(file_counts),
            "id_counts": {
                "grammar": len(grammar_ids),
                "style": len(style_ids),
                "all_unique": len(all_ids),
            },
            "ids_per_file": dict(sorted(file_counts.items())),
        }
        man_path = assets_dir / "languagetool_rule_inventory_manifest.json"
        with man_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("Wrote", man_path.relative_to(project_root))

    if args.bulk_fill_mapping:
        map_path = assets_dir / "languagetool_to_mistaketype.json"
        if not map_path.exists():
            print(f"Missing {map_path}; cannot bulk-fill.", file=sys.stderr)
            sys.exit(1)
        with map_path.open("r", encoding="utf-8") as f:
            mapping: dict[str, str] = json.load(f)
        added = 0
        for rid in sorted(all_ids):
            if rid not in mapping:
                mapping[rid] = args.default_new
                added += 1
        with map_path.open("w", encoding="utf-8") as f:
            json.dump(dict(sorted(mapping.items())), f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Bulk-fill: added {added} keys with mistake_type={args.default_new!r}")

    print("Language: en (+ regional en-* XMLs under en/)")
    print(f"Rules directory: {en_root}")
    print(f"Grammar rule IDs: {len(grammar_ids)}")
    print(f"Style rule IDs: {len(style_ids)}")
    print(f"Total unique rule IDs: {len(all_ids)}")
    print("Assets written to:", assets_dir)


if __name__ == "__main__":
    main()
