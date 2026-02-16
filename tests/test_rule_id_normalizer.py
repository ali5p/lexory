"""Tests for rule_id normalizer."""
import pytest

from rag.utils.rule_id_normalizer import normalize_rule_id


def test_normalize_base_form():
    assert normalize_rule_id("BASE_FORM") == "BASE_FORM"


def test_normalize_strips_sub_rule_suffix():
    assert normalize_rule_id("TOT_HE[1]") == "TOT_HE"
    assert normalize_rule_id("MORFOLOGIK_RULE_EN_US[42]") == "MORFOLOGIK_RULE_EN_US"


def test_normalize_uppercases():
    assert normalize_rule_id("base_form") == "BASE_FORM"


def test_normalize_empty_and_none():
    assert normalize_rule_id(None) == ""
    assert normalize_rule_id("") == ""
    assert normalize_rule_id("   ") == ""


def test_normalize_preserves_valid_chars():
    assert normalize_rule_id("EN_A_VS_AN") == "EN_A_VS_AN"
