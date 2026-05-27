#!/usr/bin/env python3
"""
Reclassify rules currently mapped to bare "confusables" into subcategories
or move them to the correct parent mistake_type when the two-token fallback
was too aggressive.

Subcategories:
  confusables.homophones  — sound-alike words (there/their, to/too/two)
  confusables.spelling    — spelling/derivation variants (-ce/-se, -ance/-ence)
  confusables.word_choice — semantically confused but distinct words

Usage:
    python scripts/subdivide_confusables.py --dry-run
    python scripts/subdivide_confusables.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = PROJECT_ROOT / "assets" / "languagetool_to_mistaketype.json"
VOCAB_PATH = PROJECT_ROOT / "assets" / "mistake_logic_vocab.json"
BACKUP_PATH = PROJECT_ROOT / "assets" / "languagetool_to_mistaketype.json.bak2"

SOURCE = "confusables"

# ---------------------------------------------------------------------------
# Explicit overrides for rules that were mis-bucketed as confusables.
# ---------------------------------------------------------------------------

EXPLICIT: dict[str, str] = {
    # User-flagged
    "WILL_BECOMING": "modal_verbs",
    "WHO_THAN": "questions",
    "WHO_WHOM": "pronouns.case",
    # Preposition / particle (not word confusion)
    "AT_ANYTIME": "prepositions.time",
    "AT_TIME": "prepositions.time",
    "BY_EXAMPLE": "prepositions",
    "BY_EXPIRE": "prepositions",
    "BY_PASS": "punctuation.hyphenation",
    "FROM_FORM": "prepositions",
    "FROM_THAN_ON": "prepositions",
    "FROM_X_Y": "prepositions",
    "ON_ADDITION": "prepositions",
    "ON_BEHAVE": "prepositions.verb",
    "ON_EXCEL": "prepositions",
    "ON_FACT": "prepositions",
    "ON_GOING": "punctuation.hyphenation",
    "ON_SKYPE": "prepositions",
    "TO_ABLE": "infinitive_vs_gerund",
    "TO_AIDE": "prepositions.verb",
    "TO_BACKOUT": "prepositions.verb",
    "TO_BATH": "prepositions.verb",
    "TO_BLACKOUT": "prepositions.verb",
    "TO_COMEBACK": "prepositions.verb",
    "TO_WITHDRAWN": "prepositions.verb",
    "WITH_OUT": "punctuation.hyphenation",
    "WITH_WIDTH": "confusables.homophones",
    # Modals
    "CAN_CAB": "modal_verbs",
    "CAN_CHECKIN": "modal_verbs",
    "MAY_MANY": "confusables.homophones",
    "MAY_MANY_MY": "confusables.homophones",
    "MUST_HAVE": "modal_verbs",
    # Tense / verb form
    "COPY_PASTE": "typos",
    "PAST_PASTE": "confusables.homophones",
    "WERE_MD": "auxiliary_verbs",
    # POS-tag rules misclassified
    "THE_CC": "sentence_structure",
    "THIS_CD": "determiners_quantifiers",
    # Genuine homophones (explicit)
    "BY_BUY": "confusables.homophones",
    "TO_TOO": "confusables.homophones",
    "TO_TWO": "confusables.homophones",
    "FOR_FRO": "confusables.homophones",
    "THERE_THEIR": "confusables.homophones",
    "THEIR_THERE": "confusables.homophones",
    "ITS_ITS": "confusables.homophones",
    "YOUR_YOURE": "confusables.homophones",
    "YOURE_YOUR": "confusables.homophones",
    "WHOSE_WHOS": "confusables.homophones",
    "WHOS_WHOSE": "confusables.homophones",
    "HEAR_HERE": "confusables.homophones",
    "HERE_HEAR": "confusables.homophones",
    "KNEW_NEW": "confusables.homophones",
    "NEW_KNEW": "confusables.homophones",
    "HOLE_WHOLE": "confusables.homophones",
    "WHOLE_HOLE": "confusables.homophones",
    "WEAK_WEEK": "confusables.homophones",
    "WEEK_WEAK": "confusables.homophones",
    "BEAN_BEEN": "confusables.homophones",
    "BEE_BEEN": "confusables.homophones",
    "BEE_BE": "confusables.homophones",
    "MEAT_MEET": "confusables.homophones",
    "MEET_MEAT": "confusables.homophones",
    "PEAK_PEEK": "confusables.homophones",
    "PEEK_PEAK": "confusables.homophones",
    "RAIN_REIGN": "confusables.homophones",
    "REIGN_RAIN": "confusables.homophones",
    "STEAL_STEEL": "confusables.homophones",
    "STEEL_STEAL": "confusables.homophones",
    "FLOUR_FLOWER": "confusables.homophones",
    "FLOWER_FLOUR": "confusables.homophones",
    "MAYOR_MAJOR": "confusables.homophones",
    "MAJOR_MAYOR": "confusables.homophones",
    "ROUTE_ROOT": "confusables.homophones",
    "ROOT_ROUTE": "confusables.homophones",
    "SAIL_SALE": "confusables.homophones",
    "SALE_SAIL": "confusables.homophones",
    "SEA_SEE": "confusables.homophones",
    "SEE_SEA": "confusables.homophones",
    "SON_SUN": "confusables.homophones",
    "SUN_SON": "confusables.homophones",
    "TAIL_TALE": "confusables.homophones",
    "TALE_TAIL": "confusables.homophones",
    "WAIT_WEIGHT": "confusables.homophones",
    "WEIGHT_WAIT": "confusables.homophones",
    "WEAR_WHERE": "confusables.homophones",
    "WHERE_WEAR": "confusables.homophones",
    "WHETHER_WEATHER": "confusables.homophones",
    "WEATHER_WHETHER": "confusables.homophones",
    "WRIGHT_RIGHT": "confusables.homophones",
    "RIGHT_WRIGHT": "confusables.homophones",
    "WRITE_RIGHT": "confusables.homophones",
    "RIGHT_WRITE": "confusables.homophones",
    "BREAK_BRAKE": "confusables.homophones",
    "BRAKE_BREAK": "confusables.homophones",
    "BARE_BEAR": "confusables.homophones",
    "BEAR_BARE": "confusables.homophones",
    "DEAR_DEER": "confusables.homophones",
    "DEER_DEAR": "confusables.homophones",
    "FAIR_FARE": "confusables.homophones",
    "FARE_FAIR": "confusables.homophones",
    "HOLE_WHOLE": "confusables.homophones",
    "HOUR_OUR": "confusables.homophones",
    "OUR_HOUR": "confusables.homophones",
    "KNOT_NOT": "confusables.homophones",
    "NOT_KNOT": "confusables.homophones",
    "LOOSE_LOSE": "confusables.homophones",
    "LOSE_LOOSE": "confusables.homophones",
    "PLAIN_PLANE": "confusables.homophones",
    "PLANE_PLAIN": "confusables.homophones",
    "PAST_PASSED": "confusables.homophones",
    "PASSED_PAST": "confusables.homophones",
    "PEACE_PIECE": "confusables.homophones",
    "PIECE_PEACE": "confusables.homophones",
    "PRINCIPAL_PRINCIPLE": "confusables.word_choice",
    "PRINCIPLE_PRINCIPAL": "confusables.word_choice",
    # Spelling / derivation pairs (explicit)
    "ADVICE_ADVISE": "confusables.spelling",
    "ADVISE_ADVICE": "confusables.spelling",
    "BELIEF_BELIEVE": "confusables.spelling",
    "BELIEVE_BELIEF": "confusables.spelling",
    "BREATH_BREATHE": "confusables.spelling",
    "BREATHE_BREATH": "confusables.spelling",
    "LICENCE_LICENSE_NOUN_PLURAL": "confusables.spelling",
    "LICENCE_LICENSE_NOUN_SINGULAR": "confusables.spelling",
    "EMPHASIS_EMPHASIZE": "confusables.spelling",
    "ESSENTIAL_ESSENTIALLY": "confusables.spelling",
    "HONESTY_HONESTLY": "confusables.spelling",
    "COMPLIMENT_COMPLEMENT": "confusables.spelling",
    "COMPLEMENT_COMPLIMENT": "confusables.spelling",
    # Relative / question words
    "WHO_THAT": "relative_clauses",
    "WHICH_THAT": "relative_clauses",
    "WHO_WHOM": "pronouns.case",
    # Short look-alike → typos
    "AD_ADD": "typos",
    "ART_ARE": "typos",
    "AS_ASK": "typos",
    "A_TO": "typos",
    "BE_IS": "typos",
    "DO_TO": "typos",
    "EGO_AGO": "typos",
    "ET_AL": "typos",
    "ET_ALL": "typos",
    "FR_FOR": "typos",
    "GOOD_WELL": "adjectives",
    "EXTREME_ADJECTIVES": "adjectives",
    "EITHER_NOR": "negation",
    "EVERY_EVER": "confusables.homophones",
    "EVENT_EVEN": "confusables.homophones",
    "ALREADY_ALL_READY": "confusables.homophones",
    "EVERYDAY_EVERY_DAY": "confusables.homophones",
    # AI confusion rules
    "AI_EN_QB_GGEC_REPLACEMENT_CONFUSION": "confusables.word_choice",
    "AI_HYDRA_LEO_CPT_THERE_THEIR": "confusables.homophones",
    "AI_HYDRA_LEO_CP_THE_THEY": "confusables.homophones",
    "AI_HYDRA_LEO_CP_THE_TO": "confusables.homophones",
    "AI_HYDRA_LEO_CP_TO_THE": "confusables.homophones",
    "AI_HYDRA_LEO_CP_YOUR_YOURE": "confusables.homophones",
    "ACCOUNTS_FOR": "prepositions.verb",
    "ALLOW_TO": "infinitive_vs_gerund",
    "ARE_OUR": "confusables.homophones",
    "AIR_HEIR": "confusables.homophones",
    "ANALYSIS_IF": "sentence_structure",
    "AS_ADJ_AS": "adjectives",
    "AFTER_BEEN": "verb_tense",
    "THAT_S_YOU_RE": "confusables.homophones",
    "THAT_THAN": "questions",
    "CONGRATULATIONS_FOR": "prepositions",
    "CONSTITUTES_OF": "prepositions",
    "CRAVE_FOR": "prepositions.verb",
    "CURATOR_OF": "prepositions",
    "DIVERSITY_OF": "prepositions",
    "CC_VERB": "sentence_structure",
    "BUNCH_OF": "prepositions",
    "CLEAN_UP": "prepositions.verb",
    "COLLEGE_COLLEAGUE": "confusables.spelling",
    "ALLY_ALLAY": "confusables.homophones",
}

# Known homophone roots (either token in pair)
_HOMOPHONE_ROOTS = frozenset({
    "there", "their", "theyre", "theyre", "they", "the",
    "to", "too", "two", "tu",
    "your", "youre", "you",
    "its", "it",
    "whos", "whose", "who", "whom",
    "hear", "here",
    "knew", "new", "gnu",
    "hole", "whole",
    "weak", "week",
    "bean", "been", "bee", "be",
    "meat", "meet",
    "peak", "peek",
    "rain", "reign", "rein",
    "steal", "steel",
    "flour", "flower",
    "mayor", "major",
    "route", "root",
    "sail", "sale",
    "sea", "see",
    "son", "sun",
    "tail", "tale",
    "wait", "weight",
    "wear", "where",
    "whether", "weather",
    "wright", "right", "write", "rite",
    "break", "brake",
    "bare", "bear",
    "dear", "deer",
    "fair", "fare",
    "hour", "our",
    "knot", "not",
    "loose", "lose",
    "plain", "plane",
    "past", "passed",
    "peace", "piece",
    "buy", "by", "bye",
    "accept", "except",
    "affect", "effect",
    "aisle", "isle",
    "allude", "elude",
    "altar", "alter",
    "breathe", "breath",
    "canvas", "canvass",
    "council", "counsel",
    "desert", "dessert",
    "device", "devise",
    "discreet", "discrete",
    "elicit", "illicit",
    "emigrate", "immigrate",
    "ensure", "insure",
    "farther", "further",
    "fewer", "less",
    "flaunt", "flout",
    "foreword", "forward",
    "imply", "infer",
    "lay", "lie",
    "lead", "led",
    "lessen", "lesson",
    "personal", "personnel",
    "precede", "proceed",
    "prescribe", "proscribe",
    "principal", "principle",
    "stationary", "stationery",
    "than", "then",
    "every", "ever",
    "event", "even",
    "already", "allready",
    "everyday", "every",
    "compliment", "complement",
    "advice", "advise",
    "belief", "believe",
    "practice", "practise",
    "licence", "license",
    "defence", "defense",
})

_PREP_PREFIXES = ("IN_", "ON_", "AT_", "FOR_", "FROM_", "BY_", "WITH_", "INTO_", "ONTO_", "ABOUT_", "ACROSS_", "AROUND_", "BETWEEN_", "AMONG_", "TOWARD_", "TOWARDS_", "UPON_", "UNDER_", "OVER_", "THROUGH_", "DURING_", "BEFORE_", "AFTER_", "SINCE_", "UNTIL_")
_PREP_PARTICLES = frozenset({
    "FOR", "TO", "IN", "ON", "AT", "OF", "BY", "AS", "IF", "OR", "AND", "UP", "OUT",
    "OFF", "OVER", "INTO", "FROM", "WITH", "ABOUT", "AFTER", "BEFORE", "UNDER",
})

_PREP_CONFUSABLES = frozenset({"TO_TOO", "TO_TWO", "FOR_FRO", "BY_BUY", "FROM_FORM"})

_MODAL_PREFIXES = ("WILL_", "WOULD_", "SHOULD_", "COULD_", "MUST_", "MAY_", "MIGHT_", "CAN_", "SHALL_")

_GRAMMAR_TAG_SUFFIXES = ("_VBN", "_VBD", "_VBZ", "_VBP", "_VB", "_VBG", "_MD", "_NN", "_NNS", "_NNP", "_PRP", "_DT", "_JJ", "_RB", "_CD", "_CC")


def _parts(rid: str) -> list[str]:
    return rid.split("_")


def _is_spelling_pair(a: str, b: str) -> bool:
    al, bl = a.lower(), b.lower()
    if (al.endswith("ce") and bl.endswith("se")) or (al.endswith("se") and bl.endswith("ce")):
        return True
    if (al.endswith("ance") and bl.endswith("ence")) or (al.endswith("ence") and bl.endswith("ance")):
        return True
    if (al.endswith("able") and bl.endswith("ible")) or (al.endswith("ible") and bl.endswith("able")):
        return True
    if (al.endswith("ise") and bl.endswith("ize")) or (al.endswith("ize") and bl.endswith("ise")):
        return True
    if (al.endswith("ly") and not bl.endswith("ly")) or (bl.endswith("ly") and not al.endswith("ly")):
        # adjective ↔ adverb derivation: quick/quickly pattern
        if abs(len(al) - len(bl)) <= 3:
            return True
    return False


def _is_homophone_pair(a: str, b: str) -> bool:
    al, bl = a.lower(), b.lower()
    if al in _HOMOPHONE_ROOTS or bl in _HOMOPHONE_ROOTS:
        return True
    return False


def classify_confusable(rid: str) -> str:
    if rid in EXPLICIT:
        return EXPLICIT[rid]

    if rid.startswith("CONFUSION_"):
        return "confusables.word_choice"

    if rid.startswith(("WHO_", "WHOM_", "WHOSE_", "WHICH_", "THAT_")) and not rid.startswith("CONFUSION_"):
        if "WHOM" in rid or rid in ("WHO_THAN", "WHO_THAT"):
            return "questions" if "THAN" in rid else "relative_clauses"
        return "relative_clauses"

    for suffix in _GRAMMAR_TAG_SUFFIXES:
        if rid.endswith(suffix):
            if suffix in ("_VB", "_VBP", "_VBZ", "_VBD", "_VBN", "_VBG"):
                return "verb_form"
            if suffix == "_MD":
                return "modal_verbs"
            if suffix in ("_NN", "_NNS"):
                return "count_uncount"
            if suffix == "_PRP":
                return "pronouns"
            if suffix == "_DT":
                return "articles.definite_indefinite"
            if suffix == "_JJ":
                return "adjectives"
            if suffix == "_RB":
                return "adverbs"
            if suffix == "_CC":
                return "sentence_structure"
            if suffix == "_CD":
                return "determiners_quantifiers"

    for prefix in _MODAL_PREFIXES:
        if rid.startswith(prefix) and rid not in _PREP_CONFUSABLES:
            return "modal_verbs"

    for prefix in _PREP_PREFIXES:
        if rid.startswith(prefix) and rid not in _PREP_CONFUSABLES:
            if prefix in ("IN_", "ON_", "AT_"):
                return "prepositions"
            if prefix == "FOR_":
                return "prepositions"
            if prefix in ("FROM_", "TO_", "BY_", "WITH_"):
                return "prepositions"
            return "prepositions"

    parts = _parts(rid)
    if len(parts) == 2:
        a, b = parts
        if _is_homophone_pair(a, b):
            return "confusables.homophones"
        if _is_spelling_pair(a, b):
            return "confusables.spelling"
        if b in _PREP_PARTICLES or a in _PREP_PARTICLES:
            if b in ("TO",) and a not in ("TO", "TOO", "TWO"):
                return "infinitive_vs_gerund"
            return "prepositions.verb" if b in ("FOR", "TO", "IN", "ON", "AT", "OF", "WITH") else "prepositions"
        # Short look-alike noise: both tokens very short
        if len(a) <= 2 and len(b) <= 2:
            return "typos"
        if max(len(a), len(b)) <= 3 and min(len(a), len(b)) <= 2:
            return "typos"
        return "confusables.word_choice"

    if len(parts) >= 3:
        return "confusables.word_choice"

    return "confusables.word_choice"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not MAPPING_PATH.exists():
        print(f"Missing {MAPPING_PATH}", file=sys.stderr)
        return 1

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)

    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocab = json.load(f).get("categories", [])
    valid = set(vocab)

    source_ids = sorted(k for k, v in mapping.items() if v == SOURCE)
    moves: dict[str, list[str]] = {}
    for rid in source_ids:
        target = classify_confusable(rid)
        if target not in valid:
            print(f"Warning: {rid} -> {target!r} not in vocab, keeping {SOURCE}", file=sys.stderr)
            target = SOURCE
        moves.setdefault(target, []).append(rid)

    counts = Counter({t: len(ids) for t, ids in moves.items()})
    print(f"Rules to reclassify from '{SOURCE}': {len(source_ids)}")
    print()
    for target, n in counts.most_common():
        sample = ", ".join(moves[target][:2])
        print(f"  {target:36s} {n:4d}   e.g. {sample}")

    if args.dry_run:
        print("\nDry run — no files modified.")
        return 0

    shutil.copy(MAPPING_PATH, BACKUP_PATH)
    for target, ids in moves.items():
        for rid in ids:
            mapping[rid] = target

    with MAPPING_PATH.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(mapping.items())), f, indent=2, ensure_ascii=False)
        f.write("\n")

    remaining = sum(1 for v in mapping.values() if v == SOURCE)
    print(f"\nBackup: {BACKUP_PATH}")
    print(f"Updated: {MAPPING_PATH}")
    print(f"Remaining bare '{SOURCE}': {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
