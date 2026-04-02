"""Unit tests for scripts/extract_languagetool_rule_ids helpers (no vendor tree required)."""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "extract_languagetool_rule_ids",
    _ROOT / "scripts" / "extract_languagetool_rule_ids.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_rule_ids_from_xml = _mod.extract_rule_ids_from_xml
classify_file = _mod.classify_file


@pytest.fixture
def sample_en_dir(tmp_path: Path) -> Path:
    en = tmp_path / "en"
    en.mkdir()
    (en / "grammar.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rules lang="en" xmlns="http://www.languagetool.org/xml">\n'
        '  <rulegroup id="GROUP_ONE">\n'
        '    <rule id="RULE_ALPHA">\n'
        "      <pattern><token>test</token></pattern>\n"
        "      <message>msg</message>\n"
        "    </rule>\n"
        "  </rulegroup>\n"
        '  <rule id="RULE_BETA"></rule>\n'
        "</rules>\n",
        encoding="utf-8",
    )
    (en / "style.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rules lang="en">\n'
        '  <rule id="STYLE_ONLY"></rule>\n'
        "</rules>\n",
        encoding="utf-8",
    )
    return en


def test_extract_collects_rule_and_rulegroup_ids(sample_en_dir: Path) -> None:
    g = extract_rule_ids_from_xml(sample_en_dir / "grammar.xml")
    assert g == {"GROUP_ONE", "RULE_ALPHA", "RULE_BETA"}


def test_classify_file_buckets(sample_en_dir: Path) -> None:
    assert classify_file(sample_en_dir, sample_en_dir / "grammar.xml") == "grammar"
    assert classify_file(sample_en_dir, sample_en_dir / "style.xml") == "style"
