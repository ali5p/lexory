#!/usr/bin/env python3
"""
Classify rule IDs currently mapped to "unmapped" into more specific
mistake_types using priority-ordered heuristics.

Rules:
  - Conservative: when no heuristic matches confidently, the rule stays
    in "unmapped".
  - Idempotent: existing non-unmapped mappings are NEVER modified.
  - Safe: writes a backup of the mapping file before applying changes.

With --apply, run `scripts/subdivide_confusables.py` afterward if any rules
land in bare "confusables" (legacy) or need finer splitting.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Pattern, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MAPPING_PATH = ASSETS_DIR / "languagetool_to_mistaketype.json"
VOCAB_PATH = ASSETS_DIR / "mistake_logic_vocab.json"
BACKUP_PATH = ASSETS_DIR / "languagetool_to_mistaketype.json.bak"
RESIDUAL_REPORT_PATH = ASSETS_DIR / "unmapped_residual.json"

UNMAPPED = "unmapped"


# ---------------------------------------------------------------------------
# Curated explicit overrides. Used when a regex would be too risky.
# These are rules whose ID alone is enough to know the type with confidence.
# ---------------------------------------------------------------------------

EXPLICIT_OVERRIDES: dict[str, str] = {
    # ---- Articles & determiners -----------------------------------------
    "A_NN": "articles.a_an",
    "A_NNS": "articles.a_an",
    "A_NNS_AND": "articles.a_an",
    "A_NNS_BEST_NN": "articles.a_an",
    "A_NUMBER_NNS": "articles.a_an",
    "A_INFINITIVE": "articles.a_an",
    "AN_VB_PRP": "articles.a_an",
    "A_MD_VB": "articles.a_an",
    "A_MUCH_NN1": "articles.a_an",
    "A_RB_A_JJ_NN": "articles.a_an",
    "A_BIT": "articles.a_an",
    "A_BIT_OF": "articles.a_an",
    "A_LOT_OF_NN": "articles.a_an",
    "A_COLLECTIVE_OF_NN": "articles.a_an",
    "ARTICLE_MISSING": "articles.definite_indefinite",
    "ARTICLE_ADJECTIVE_OF": "articles.definite_indefinite",
    # ---- Count / uncount ------------------------------------------------
    "A_INFORMATION": "count_uncount",
    "A_FEEDBACK": "count_uncount",
    "A_UNCOUNTABLE": "count_uncount",
    "A_SCISSOR": "count_uncount",
    "AIRCRAFTS": "count_uncount",
    # ---- Pronouns -------------------------------------------------------
    "AGREEMENT_THEIR_HIS": "pronouns",
    "CONFUSION_OF_ME_I": "pronouns.case",
    "ME_VS_I": "pronouns.case",
    "I_VS_ME": "pronouns.case",
    "WHO_VS_WHOM": "pronouns.case",
    "WHO_WHOM": "pronouns.case",
    # ---- Verb form / aux / gerund / tense -------------------------------
    "ABOUT_TO_VBD": "verb_form",
    "ADMIT_ENJOY_VB": "verb_form",
    "ADVISE_VBG": "verb_form",
    "AFFORD_VB": "infinitive_vs_gerund",
    "AFFORD_VBG": "infinitive_vs_gerund",
    "AVOIDING_TO_INFIN": "infinitive_vs_gerund",
    "BARE_INFINITIVE_VERB_PRP_VBZ": "verb_form",
    "BASE_FORM": "verb_form",
    "BARE_IN_MIND": "idioms",
    "AUXILIARY_DO_WITH_INCORRECT_VERB_FORM": "auxiliary_verbs",
    "BEEN_PART_AGREEMENT": "subject_verb_agreement",
    "BE_PART_AGREEMENT_2": "subject_verb_agreement",
    "AS_WELL_AS_AGREEMENT": "subject_verb_agreement",
    # ---- Word order -----------------------------------------------------
    "ADVERB_WORD_ORDER": "word_order",
    "ADVERB_WORD_ORDER_10_TEMP": "word_order",
    # ---- Adjective vs adverb -------------------------------------------
    "ADJECTIVE_ADVERB": "adjectives",
    "ADJECTIVE_ADVERB_2": "adjectives",
    # ---- Punctuation ----------------------------------------------------
    "ABBREVIATION_PUNCTUATION": "punctuation.basic",
    "AM_PM": "punctuation.basic",
    "AM_PM_OCLOCK": "punctuation.basic",
    "AT_CD_CLOCK": "punctuation.basic",
    # ---- Prepositions (verb collocations) -------------------------------
    "ACCOMPANY_WITH": "prepositions.verb",
    "ACCUSE_FOR_OFF": "prepositions.verb",
    "ADOPT_TO": "prepositions.verb",
    "ASSOCIATES_TO": "prepositions.verb",
    "ASSOCIATE_TOGETHER": "prepositions.verb",
    "BORN_IN": "prepositions.verb",
    "BUILD_OFF_OF": "prepositions.verb",
    "ARRIVAL_TO_THE_HOUSE": "prepositions.noun",
    # ---- Confusables (high-confidence pairs) ----------------------------
    "ACCEPT_EXCEPT": "confusables.homophones",
    "ACCESS_EXCESS": "confusables.word_choice",
    "ACTIVE_ACTIVATE": "confusables.word_choice",
    "ADDEND_ATTEND": "confusables.word_choice",
    "ADMIN_ADMIT": "confusables.word_choice",
    "ADVERSE_AVERSE": "confusables.homophones",
    "ADVICE_ADVISE": "confusables.spelling",
    "AFFECT_EFFECT": "confusables.homophones",
    "AFFECTED_EFFECTED": "confusables.homophones",
    "AFFECTS": "confusables.homophones",
    "AIR_HEIR": "confusables.homophones",
    "AIRPLANE_HANGER": "confusables.homophones",
    "AISLE_ISLE": "confusables.homophones",
    "ALLUDE_ELUDE": "confusables.homophones",
    "ALLY_ALLAY": "confusables.homophones",
    "ALTER_ALTAR": "confusables.homophones",
    "AMASSING_AMAZING": "confusables.word_choice",
    "AMENABLE_AMENDABLE": "confusables.spelling",
    "ANUS_ANGUS": "confusables.homophones",
    "APART_A_PART": "confusables.homophones",
    "APART_FORM": "confusables.word_choice",
    "APOSTROPHE_VS_QUOTE": "punctuation.basic",
    "ASCETIC_ACID": "confusables.word_choice",
    "ASSES_ASSESS": "confusables.word_choice",
    "ATTACHE_ATTACH": "confusables.word_choice",
    "AXED_ASKED": "confusables.homophones",
    "BARLEY_BARELY": "confusables.spelling",
    "BATTER_BETTER": "confusables.homophones",
    "BELIEF_BELIEVE": "confusables.spelling",
    "BELIEVE_BELIEF": "confusables.spelling",
    "BELIVE_BELIEVE": "typos",
    "BENTS_BENDS": "confusables.homophones",
    "BREATHE_BREATH": "confusables.spelling",
    "BREATH_OF_SCOPE_BREADTH_OF_SCOPE": "confusables.word_choice",
    "BROWS_BROWSE": "confusables.word_choice",
    "BELOW_BLOW": "confusables.homophones",
    "BEAN_BEEN": "typos",
    "BEE_BE": "typos",
    "BEE_BEEN": "typos",
    # ---- Typos (single-letter / look-alike) -----------------------------
    "AD_ADD": "typos",
    "AD_AND": "typos",
    "AD_AS": "typos",
    "AD_NAUSEUM": "typos",
    "AFTERALL": "typos",
    "AFTER_NOON": "typos",
    "AIR_BORNE_AIRBORN": "typos",
    "AL_ALL": "typos",
    "ALINE": "typos",
    "ALLTHOUGH": "typos",
    "AM_I": "typos",
    "AND_BUT": "typos",
    "AND_END": "typos",
    "AND_THAN": "typos",
    "ANI_T": "typos",
    "ANS_AND": "typos",
    "ANTHER": "typos",
    "AN_AND": "typos",
    "AN_THEN": "typos",
    "APOS_ARE": "typos",
    "APOS_RE": "typos",
    "APPRECIATE_IF": "style",
    "ARN_T": "typos",
    "ART_ARE": "typos",
    "ASS_AS": "typos",
    "AS_ASK": "typos",
    "AS_DISCUSS": "typos",
    "AS_FOLLOW": "typos",
    "AS_FOLLOW_AS_FOLLOWS": "typos",
    "AS_MENTION": "typos",
    "AS_OPPOSE_TO_AS_OPPOSED_TO": "typos",
    "AS_SAD": "typos",
    "AS_US": "typos",
    "AS_WILL_AS": "typos",
    "AT_AS": "typos",
    "AVE_HAVE": "typos",
    "AWAY_FRO": "typos",
    "A_TO": "typos",
    "A_BUT": "typos",
    "A_OR": "typos",
    "A_WAS": "typos",
    "A_MY": "typos",
    "BE_CAUSE": "typos",
    "BE_COME": "typos",
    "BE_DONT": "typos",
    "BE_HAVENT": "typos",
    "BE_IS": "typos",
    "BE_NO_VB": "typos",
    "BE_SEEM": "typos",
    "BE_VBP_IN": "typos",
    "BE_WARE": "typos",
    "BE_WILL": "typos",
    "BEFORE_HAND": "typos",
    "BEG_BAG": "typos",
    "BET_BEST": "typos",
    "BIS_BUS": "typos",
    "BOB_WIRE": "typos",
    "BON_APPETITE": "typos",
    "BOUT_TO": "typos",
    "BRAN_BRAND": "typos",
    "BREW_HAHA": "typos",
    "BUT_VERSUS_AND": "typos",
    "BU": "typos",
    "BYE_THE_WAY": "typos",
    "BY_BUY": "confusables.homophones",
    # ---- Idioms (fixed-expression corrections) --------------------------
    "ALL_OF_A_SUDDEN": "idioms",
    "ALL_OF_THE_SUDDEN": "idioms",
    "ALONG_THE_SAME_VEIN": "idioms",
    "ALONG_TIME": "idioms",
    "ALREADY_ALL_READY": "confusables.homophones",
    "AND_SO_FOURTH": "idioms",
    "AND_SO_ONE": "idioms",
    "BACK_AND_FOURTH": "idioms",
    "BACK_IN_FORTH": "idioms",
    "BAITED_BREATH": "idioms",
    "BARE_IN_MIND_2": "idioms",
    "BEAT_REGARDS": "idioms",
    "BECKON_CALL": "idioms",
    "BEGS_BELIEF": "idioms",
    "BESIDES_THE_POINT": "idioms",
    "BETTER_SAFE_THAN_SORRY": "idioms",
    "BEYOND_THE_PAIL": "idioms",
    "BREAKING_GOOD": "idioms",
    "BRING_PUN_ON_THE_AGENDA": "idioms",
    # ---- Style (redundancy / wordiness) ---------------------------------
    "ABOVE_MENTIONED": "style",
    "ADDITIONAL": "style",
    "ADD_AN_ADDITIONAL": "style",
    "ADD_NO": "style",
    "ACCORDING_TO_ME": "style",
    "AGREE_WITH_THE_FACT": "style",
    "DESPITE_THE_FACT": "style",
    "EXACT_SAME": "style",
    "EVERYONE_OF": "style",
    "FORMER_ALUMNUS": "style",
    "FORMALLY_KNOWN_AS": "style",
    "INDUSTRY_LEADING_NN": "style",
    "MARRIAGE_ANNIVERSARY": "style",
    # ---- Verb form (past-tense / participle errors) ---------------------
    "AWAKED": "verb_form",
    "BENDED": "verb_form",
    "BIDDED": "verb_form",
    "CASTED": "verb_form",
    "CHOOSED": "verb_form",
    "GETTED": "verb_form",
    "GET_VBN": "verb_form",
    "GOT_GOTTEN": "verb_form",
    "JAILBREAKED": "verb_form",
    "DID_BASEFORM": "verb_form",
    "DID_PAST": "verb_tense",
    "IRREGULAR_PAST_PARTICIPLES": "verb_form",
    # ---- Sentence structure ---------------------------------------------
    "MISSING_NOUN": "sentence_structure",
    "MISSING_VERB_AFTER_WHAT": "sentence_structure",
    "COLLECTIVE_NOUN_VERB_AGREEMENT_VBD": "subject_verb_agreement",
    "COMPOUND_ADJECTIVE_NOUN": "punctuation.hyphenation",
    # ---- Determiners / quantifiers --------------------------------------
    "EACH_EVERY_NNS": "determiners_quantifiers",
    "MANY_NN_U": "determiners_quantifiers",
    "FEWER_LESS": "determiners_quantifiers",
    "HUNDREDS_OF_THOUSAND": "determiners_quantifiers",
    "HUNDRED_OF_PLURAL": "determiners_quantifiers",
    "LARGE_NUMBER_OF": "determiners_quantifiers",
    "DT_DT": "articles.definite_indefinite",
    "DT_JJ_NO_NOUN": "articles.definite_indefinite",
    "DT_RESPONDS": "articles.definite_indefinite",
    "DETERMINER_GEOGRAPHICAL_WORD": "articles.definite_indefinite",
    # ---- Verb tense / form ----------------------------------------------
    "DATE_FUTURE_VERB_PAST": "verb_tense",
    "DATE_FUTURE_VERB_PAST_US": "verb_tense",
    "DID_A_MISTAKE": "verb_form",
    "DIDN_T_BEEN_SOLVED": "verb_form",
    "DIDN_T_SAW": "verb_form",
    "DIDN_T_SPOKE": "verb_form",
    "DID_FOUND_AMBIGUOUS": "verb_form",
    "DID_FOUND_AMBIGUOUS_2": "verb_form",
    "DID_YOU_HAVE_VBN": "verb_form",
    "DOES_X_HAS": "verb_form",
    "DOES_XX_CAN": "verb_form",
    "DOES_YOU": "subject_verb_agreement",
    "DO_HE_VERB": "subject_verb_agreement",
    "GOOD_EDUCATED": "verb_form",
    "GOING_TO_VACATION": "infinitive_vs_gerund",
    "GOING_TO_JJ": "verb_form",
    "GOING_BE": "verb_form",
    "GOOD_WELL": "adjectives",
    "HAVE_BIN": "verb_form",
    "HAVE_HAVING": "verb_form",
    "HAVE_HAVE": "verb_form",
    "HAVE_NO_VB": "verb_form",
    "HAVE_RB_HAVE": "verb_form",
    "HAVE_A_VBN": "verb_form",
    "HAVE_FOLLOWING_NN": "verb_form",
    "HAS_OUGHT": "modal_verbs",
    "HAS_TO_APPROVED_BY": "passive_voice",
    "HAS_IT_NNS__IT_ITS": "subject_verb_agreement",
    "HE_VERB_AGR": "subject_verb_agreement",
    "HOW_DOES_THIS_CHANGES": "subject_verb_agreement",
    "HOW_IS_ARE": "subject_verb_agreement",
    "HOW_DO_I_VB": "questions",
    "IS_AND_ARE": "subject_verb_agreement",
    "IS_VBZ": "subject_verb_agreement",
    "IS_WAS": "verb_tense",
    "IS_WERE": "subject_verb_agreement",
    "IS_RB_BE": "verb_form",
    "IS_EVEN_WORST": "comparatives_superlatives",
    "GETS_WORST": "comparatives_superlatives",
    # ---- Capitalization / orthography ----------------------------------
    "CAPITALIZATION": "punctuation.basic",
    "CAPITALIZATION_NNP_DERIVED": "punctuation.basic",
    "EUCLIDEAN_CAPITALIZATION": "punctuation.basic",
    "ID_CASING": "punctuation.basic",
    "I_LOWERCASE": "punctuation.basic",
    "LOWERCASE_MONTHS": "punctuation.basic",
    "LOWERCASE_NAMES": "punctuation.basic",
    "FILE_EXTENSIONS_CASE": "punctuation.basic",
    "EN_QUOTES": "punctuation.basic",
    "GERMAN_QUOTES": "punctuation.basic",
    "LIGATURES": "punctuation.basic",
    "EG_SPACE": "punctuation.basic",
    "CURRENCY": "punctuation.basic",
    "CURRENCY_SPACE": "punctuation.basic",
    "CONSECUTIVE_SPACES": "punctuation.basic",
    "CONSECUTIVE_SPACES_SINGLE_CHAR": "punctuation.basic",
    "DELETE_SPACE": "punctuation.basic",
    "CONTRACTION_CASE": "punctuation.basic",
    "INCORRECT_CONTRACTIONS": "punctuation.basic",
    "IVE_CONTRACTION": "punctuation.basic",
    "POSSESSIVE_CASE": "punctuation.basic",
    "EN_MASS": "punctuation.basic",
    # ---- Pronouns -------------------------------------------------------
    "EMPHATIC_REFLEXIVE_PRONOUNS": "pronouns.reflexive",
    "EM_ME": "pronouns.case",
    "HIS_HE_S": "pronouns.possessive",
    "HE_AND_ME": "pronouns.case",
    "HE_HIS": "pronouns.case",
    "HER_HEAR": "pronouns",
    "HER_HERE": "pronouns",
    "HER_S": "pronouns.possessive",
    "I_ME": "pronouns.case",
    "ITS_TO_IT_S": "pronouns.possessive",
    "IT_ITS": "pronouns.possessive",
    "IT_SELF": "pronouns.reflexive",
    "MISSING_GENITIVE": "pronouns.possessive",
    "INCORRECT_POSSESSIVE_APOSTROPHE": "pronouns.possessive",
    # ---- Idioms (fixed-expression / wrong-word in stock phrase) --------
    "CASE_AND_POINT": "idioms",
    "CHALK_FULL": "idioms",
    "CHOMPING_AT_THE_BIT": "idioms",
    "CLOSE_SCRUTINY": "idioms",
    "COMPRISED_OF": "idioms",
    "CONSTELLATION_PRIZE": "idioms",
    "CONSTRUCTION_SIGHT": "idioms",
    "CURSING_THROUGH_VEINS": "idioms",
    "DAILY_REGIMENT": "idioms",
    "DAMP_SQUID": "idioms",
    "DEATH_NAIL": "idioms",
    "DEEP_SEEDED": "idioms",
    "DESPITE_OF": "idioms",
    "DIED_IN_THE_WOOL": "idioms",
    "DIFFERENT_TACT": "idioms",
    "DOG_EAT_DOG": "idioms",
    "FAR_BE_IT_FOR_ME": "idioms",
    "FOR_ALL_INTENSIVE_PURPOSES": "idioms",
    "FOWL_SWOOP": "idioms",
    "FREE_REIGN": "idioms",
    "FROM_THE_GET_GO": "idioms",
    "FURTHER_ADIEU": "idioms",
    "GRASPING_FOR_STRAWS": "idioms",
    "HARDLY_NEVER": "idioms",
    "HEW_AND_CRY": "idioms",
    "HIT_THE_BREAKS": "idioms",
    "IN_VEIN": "idioms",
    "INTENSIVE_PURPOSES": "idioms",
    "LAID_AHEAD": "idioms",
    "LAUGHING_STOCK": "idioms",
    "LEAD_ROLL": "idioms",
    "MAKE_SINCE": "idioms",
    "MANAGERIAL_REIGNS": "idioms",
    "AT_THE_REIGNS": "idioms",
    "MONEY_IS_NO_OPTION": "idioms",
    # ---- Typos (one-word vs two-word, contractions, misspellings) -------
    "AFTERMARKET": "typos",
    "ALONG_SIDE": "typos",
    "ALSO_KNOW": "typos",
    "ANOTHER_WORDS": "typos",
    "ANYMORE": "typos",
    "ANY_BODY": "typos",
    "APARTMENT-FLAT": "typos",
    "ASIDE": "typos",
    "ASK_WETHER": "typos",
    "BACHELORS": "typos",
    "BEFORE_HAND": "typos",
    "BESIDES_BESIDE": "typos",
    "CALENDER": "typos",
    "CANN_T": "typos",
    "CANT": "typos",
    "COLDN_T": "typos",
    "COUD_T": "typos",
    "COUN_T": "typos",
    "DAYTIME": "typos",
    "DIDENT": "typos",
    "DIDINT": "typos",
    "DOB_T": "typos",
    "DOESENT": "typos",
    "DONN_T": "typos",
    "DONS_T": "typos",
    "DONTCHA": "typos",
    "DONT_T": "typos",
    "DOSNT": "typos",
    "DOWNPAYMENT": "typos",
    "DUNKIN_DONUTS": "unmapped",
    "DUNNO": "typos",
    "EGGPLANT": "typos",
    "EVERYDAY_EVERY_DAY": "typos",
    "FREE_LANCE": "typos",
    "FREE_LANCER": "typos",
    "FREE_LANCES": "typos",
    "FREE_LANCING": "typos",
    "FRENCH_S": "typos",
    "FULL_TIME": "typos",
    "FEED_BACK": "typos",
    "FLAG_SHIP": "typos",
    "GONNA": "typos",
    "GONNA_TEMP": "typos",
    "GOTTA": "typos",
    "GOON": "typos",
    "G_MAIL": "typos",
    "HALLOWEEN": "unmapped",
    "HAVNT": "typos",
    "HAY_DAY": "typos",
    "HEAVY_WEIGHT": "typos",
    "HIGH_LIGHT": "typos",
    "HOW_EVER": "typos",
    "I_M_A_M": "typos",
    "INCASE_OF": "typos",
    "ISEN_T": "typos",
    "IDK": "typos",
    "I_AM_NOTE_SURE": "typos",
    "JAVA_SCRIPT": "typos",
    "KEY_WORDS": "typos",
    "LIGHTYEAR": "typos",
    "LIKELY_HOOD": "typos",
    "LINKED_IN": "typos",
    "LINKEDIN": "unmapped",
    "LOG_IN": "typos",
    "LOT_S": "typos",
    "MEAN_WHILE": "typos",
    "MAY_BE": "typos",
    "MASTERS": "typos",
    "MISS_SPELLING": "typos",
    # ---- Stays unmapped (proper names, brands, cultural refs) -----------
    "ALA_MODE": "unmapped",
    "ALKA_SELTZER": "unmapped",
    "AMERICANO": "unmapped",
    "APPLE_A_DAY": "unmapped",
    "APPLE_PRODUCTS": "unmapped",
    "APPSTORE": "unmapped",
    "APRIL_FOOLS": "unmapped",
    "AREA_51": "unmapped",
    "ASSASSINS_CREED": "unmapped",
    "ATM_MACHINE": "unmapped",
    "BARACK_OBAMA": "unmapped",
    "BAYERN": "unmapped",
    "BAY_AREA": "unmapped",
    "BLACK_LIVES_MATTER": "unmapped",
    "BLACK_SEA": "unmapped",
    "BLU_RAY": "unmapped",
    "BUENOS_DIAS": "unmapped",
    "CALL_OF_DUTY": "unmapped",
    "CARNEGIE_MELLON": "unmapped",
    "CHAT_GPT": "unmapped",
    "CHRISTMAS": "unmapped",
    "CHRISTMAS_TIME": "unmapped",
    "COVID_19": "unmapped",
    "EDGAR_ALLAN_POE": "unmapped",
    "EIFFEL_TOWER": "unmapped",
    "FEDEX": "unmapped",
    "FORREST_GUMP": "unmapped",
    "FOX_NEWS": "unmapped",
    "GAMEBOY": "unmapped",
    "GEIGER_COUNTER": "unmapped",
    "GINI_COEFFICIENT": "unmapped",
    "GITHUB": "unmapped",
    "GIT_HUB": "unmapped",
    "GOOGLE_PRODUCTS": "unmapped",
    "GOOGLE_PRODUCTS_STYLE": "unmapped",
    "GREYS_ANATOMY": "unmapped",
    "HAPPY_EASTER": "unmapped",
    "HARRISON_FORD": "unmapped",
    "HAWAIIAN": "unmapped",
    "HIPAA": "unmapped",
    "HOMO_ERECTUS": "unmapped",
    "HOMO_SAPIENS": "unmapped",
    "JAPAN": "unmapped",
    "JOHNS_HOPKINS_UNIVERSITY": "unmapped",
    "JONG_UN": "unmapped",
    "KAMA_SUTRA": "unmapped",
    "KELLOGGS": "unmapped",
    "KINGS_COLLEGE": "unmapped",
    "LA_PAZ": "unmapped",
    "LCD_DISPLAY": "unmapped",
    "LEHMANN_BROTHERS": "unmapped",
    "LEROY_SANE": "unmapped",
    "LOCKHEED_MARTIN": "unmapped",
    "LOREAL": "unmapped",
    "LONG_ISLAND_ICED_TEA": "unmapped",
    "MAC_OS": "unmapped",
    "MAR_A_LAGO": "unmapped",
    "MCDONALDS": "unmapped",
    "MERCEDES_BENZ": "unmapped",
    "MERRIAM_WEBSTER": "unmapped",
    "MICROSOFT_PRODUCTS": "unmapped",
    "MICROSOFT_PRODUCTS_STYLE": "unmapped",
    "MOLOTOV_COCKTAIL": "unmapped",
    "MOORES_LAW": "unmapped",
    "MOTHER-IN-LOW": "typos",
    "MUSN_T": "typos",
    "NDRANGHETA": "unmapped",
    "NIP_IT_IN_THE_BUTT": "idioms",
    "NOT_AD_ALL": "typos",
    "NOW_AND_DAYS": "idioms",
    "NOW_A_DAYS": "typos",
    "NO_COMMA_BEFORE_SO": "punctuation.basic",
    "NO_DET_NOUN_OF": "articles.definite_indefinite",
    "OBJECTIVE_C": "unmapped",
    "ODNT": "typos",
    "OK": "unmapped",
    "OM": "unmapped",
    "ONBOARD": "typos",
    "ONCE_AND_A_WHILE": "idioms",
    "ONES": "unmapped",
    "ONE_FELL_SWOOP": "idioms",
    "ONE_IN_THE_SAME": "idioms",
    "ONE_OF_THE_KIND": "idioms",
    "ONE_OF_THE_ONLY": "idioms",
    "ONE_THE_ONE_HAND": "idioms",
    "ONE_YEARS_OLD": "subject_verb_agreement",
    "ONTO_ON_TO": "typos",
    "ON-GOING": "typos",
    "ON_THE_LAMB": "idioms",
    "ON_THE_LOOK_OUT": "idioms",
    "ON_THE_CONTRAIRE": "idioms",
    "ON_THE_SAME_TOKEN": "idioms",
    "OUT_OF_SINK": "idioms",
    "OUT_OF_PLACE": "idioms",
    "OUT_OF_THE_WAY": "idioms",
    "OVERNIGHT": "typos",
    "PAPUA_NEW_GUINEA": "unmapped",
    "PARMESAN": "unmapped",
    "PAYED": "typos",
    "PAYPAL": "unmapped",
    "PEAK_HIS_INTEREST": "idioms",
    "PEAS_IN_A_POT": "idioms",
    "PEDAL_TO_THE_MEDAL": "idioms",
    "PEDAL_TO_THE_METAL": "idioms",
    "PERFECT_TENSE_SINCE": "verb_tense",
    "PERSONA_NON_GRATA": "unmapped",
    "PERS_PRON_CONTRACTION": "punctuation.basic",
    "PH_D": "unmapped",
    "PILATES": "unmapped",
    "PLAYS_A_FACTOR": "idioms",
    "PLEASE_TO_INFORM": "idioms",
    "PLEASE_TO_MEET_YOU": "idioms",
    "PLURAL_VERB_AFTER_THIS": "subject_verb_agreement",
    "PUBIC_X": "typos",
    "REAP_WHAT_YOU_SEW": "idioms",
    "REASON_IS_BECAUSE": "style",
    "REIGNS_OF_POWER": "idioms",
    "RETURN_IN_THE": "prepositions",
    "RIGHT_OF_PASSAGE": "idioms",
    "RING_ITS_NECK": "idioms",
    "ROADBLOCK": "typos",
    "ROYAL_AIR_FORCE": "unmapped",
    "SAAS": "unmapped",
    "SAFETY_DEPOSIT_BOX": "idioms",
    "SAINSBURYS": "unmapped",
    "SALT_TO_INJURY": "idioms",
    "SCHROEDINGER": "unmapped",
    "SHOWED_SHOWN_22": "verb_form",
    "SHUTDOWN": "typos",
    "SIMILAR_LIKE_AS_SOMETHING": "idioms",
    "SINGULAR_NOUN_THAT_PLURAL_VERB": "subject_verb_agreement",
    "SINGULAR_VERB_AFTER_THESE_OR_THOSE": "subject_verb_agreement",
    "SIT_ON_THE_COURT": "idioms",
    "MOVIE_THEATER_CINEMA": "unmapped",
    "MORE_EASY_N_CLEAR": "comparatives_superlatives",
    "NEEDNT_TO_DO_AND_DONT_NEED_DO": "modal_verbs",
    "REALMS_OF_POSSIBILITY": "idioms",
    "FAIRED_AS_WELL": "idioms",
    "FAIRED_BADLY": "idioms",
    "I_AM_NOTE_SURE": "typos",
    "ILL_I_LL": "typos",
    "IM_I_M": "typos",
    "I_A_M": "typos",
    "I_VE_A": "typos",
    "I_M_MD": "typos",
    "MAY_BE": "typos",
    "JAPANISE_JAPANESE": "typos",
    "JIVE_WITH": "idioms",
    "GROUND_VS_MINCED": "unmapped",
    "FALL_IS_AMONG": "verb_tense",
    "FAST_PACED_PASTE": "punctuation.hyphenation",
    "FOR_VB": "prepositions.verb",
    "FOOD_BORNE_FOODBORN": "typos",
    "FROM_THAN_ON": "typos",
    "FROM_X_Y": "prepositions",
    "FULL_COMPLIMENT_OF": "confusables.word_choice",
    "GET_A_JOB_IN_WITH": "prepositions.verb",
    "GO_TO_HOME": "prepositions",
    "HAND_AND_HAND": "idioms",
    "HAPPY_TO_YOUR": "prepositions",
    "HAVE_A_BREAKFAST": "articles.a_an",
    "HAVE_A_LOOK": "articles.a_an",
    "HAVE_CD_YEARS": "verb_form",
    "HAVE_TWITTER": "verb_form",
    "HEAVE_USE_OF": "idioms",
    "HOW_TO_IT": "sentence_structure",
    "I_NEVER_HAVE_BEEN": "verb_tense",
    "I_AS_LOOKING": "verb_tense",
    "I_BORN": "passive_voice",
    "I_AFRAID": "verb_form",
    "I_AM_VB": "verb_form",
    "I_AM_WORRY": "verb_form",
    "I_FEEL": "verb_form",
    "I_NOT_JJ": "negation",
    "I_NO_GOOD": "negation",
    "I_DONT_DT": "articles.definite_indefinite",
    "I_PERSONAL": "verb_form",
    "I_WANDER_IF": "typos",
    "I_THIN": "typos",
    "I_HERD": "typos",
    "FROM_THE_GET_GO": "idioms",
    "GOOD_IN_AT": "prepositions.adjective",
    "GOT_IT_DONE": "verb_form",
    "EXCEPTION_OF_TO": "prepositions",
    "EVERYTHING_WENT_GOOD": "adjectives",
    "EVER_SO_OFTEN": "idioms",
    "ESPECIALLY": "style",
    "EASE_OFF_USE": "idioms",
    "EASIEST_WAS_TO": "verb_tense",
    "FED_UP_OF_WITH": "prepositions.adjective",
    "FEEL_TREE_TO": "idioms",
    "FIRSTLY_OF_ALL": "idioms",
    "DARING-DO": "typos",
    "DOG-EAT-DOG": "punctuation.hyphenation",
    "DOS_AND_DONTS": "punctuation.basic",
    "DROP_IN_NN": "articles.definite_indefinite",
    "EVERY_EVER": "confusables.word_choice",
    "EVER_DAY": "typos",
    "EVENT_EVEN": "confusables.word_choice",
    "EVEN_THOU": "typos",
    "EVEN_ALTHOUGH": "style",
    "EQUALLY_AS": "style",
    "DENIAL_OF_SERVICE_ATTACK": "unmapped",
    "DEPENDENT": "typos",
    "DE_JURE_DU_JOUR": "idioms",
    "DEPEND_ON": "prepositions.verb",
    "EAT_ANTIBIOTICS": "confusables.word_choice",
    "ETHER_EITHER": "confusables.word_choice",
    "ER": "unmapped",
    "FR": "unmapped",
    "MI": "unmapped",
    "MAH": "unmapped",
    "NE": "unmapped",
    "NEE": "unmapped",
    "NOW": "unmapped",
    "HEP": "unmapped",
    "HING": "unmapped",
    "LING": "unmapped",
    "LED": "unmapped",
    "BU": "typos",
    "AU": "unmapped",
    "AI": "unmapped",
    "AIT": "unmapped",
    "DOS": "unmapped",
    "AS_ADJ_AS": "adjectives",
    "AS_TIME_PROGRESSED": "verb_tense",
    "AS_WELL_AS_BETTER": "comparatives_superlatives",
    "ATD_VERBS_TO_COLLOCATION": "prepositions.verb",
    "AT_THE_JOB": "prepositions",
    "ARE_ALLOWED_TO": "passive_voice",
    "ARE_STILL_THE_SOME": "sentence_structure",
    "ARE_WE_HAVE": "subject_verb_agreement",
    "ARROWS": "unmapped",
    "AM_LOATHE_TO": "idioms",
    "AM_IN_THE_MORNING": "punctuation.basic",
    "ANINFOR_EVERY_DAY": "typos",
    "AIR_PLANE_AEROPLANE": "typos",
    "ABOUT_ITS_NN": "pronouns.possessive",
    "AFRAID_OF_HEIGHT": "count_uncount",
    "ALONG": "prepositions",
    "A_ATTACH": "articles.a_an",
    "A_CAPPELLA": "unmapped",
    "A_CD_NNS": "articles.a_an",
    "A_COMPLAIN": "articles.a_an",
    "A_DISCOVER": "articles.a_an",
    "A_GOOGLE": "articles.a_an",
    "A_HASTILY_WAY": "articles.a_an",
    "A_HEADS_UP": "articles.a_an",
    "A_HUNDREDS": "articles.a_an",
    "A_INSTALL": "articles.a_an",
    "A_KNOW_BUG": "articles.a_an",
    "A_LA_DIACRITIC": "unmapped",
    "A_LONG": "articles.a_an",
    "A_OK": "articles.a_an",
    "A_QUITE_WHILE": "articles.a_an",
    "A_SNICKERS": "articles.a_an",
    "A_STOKE": "articles.a_an",
    "A_THANK_YOU": "articles.a_an",
    "A_TRIP_TO": "articles.a_an",
    "A_WINDOWS": "articles.a_an",
    "AI_EN_G_INSERT_OTHER": "unmapped",
    "AI_EN_G_REPLACE_ORTH": "typos",
    "AI_EN_G_REPLACE_OTHER": "unmapped",
    "AI_EN_G_REPLACE_VERB": "verb_form",
    "AI_EN_QB_GGEC_REPLACEMENT_OTHER": "unmapped",
    "BAND-AID_PLASTER": "unmapped",
    "BAR_B_QUE": "unmapped",
    "BECAUSE": "typos",
    "BECAUSE_OF_I": "pronouns.case",
    "BEGINNING_TO_ADDING_BROAD": "verb_form",
    "BEGINNING_TO_ADDING_NARROW": "verb_form",
    "BETWEEN_TO_AND": "prepositions",
    "BLOOD_BORNE_BLOODBORNE": "typos",
    "BRAKE_AWAY_BREAK_AWAY": "confusables.word_choice",
    "BUTTLOAD": "style",
    "CANT_HELP_BUT": "idioms",
    "CD_00_O_CLOCK": "punctuation.basic",
    "CD_DAY_WEEK": "punctuation.basic",
    "CD_DOZENS_OF": "determiners_quantifiers",
    "CD_OF_MY_FRIEND": "determiners_quantifiers",
    "CD_WEEK_S": "punctuation.basic",
    "CELSIUS": "unmapped",
    "CLICK_THROUGH_RATE": "unmapped",
    "COLLOCATION_ERRORS_BOKOMARU": "unmapped",
    "COME_IN_CAR": "prepositions",
    "COME_TO_PLANE": "prepositions",
    "COMPERE": "typos",
    "COPD": "unmapped",
    "COPYRIGHT": "unmapped",
    "COTE_D_AZUR": "unmapped",
    "COUPLE_OF_TIMES": "determiners_quantifiers",
    "DATE_NEW_YEAR": "punctuation.basic",
    "DIFFICULT_TO_ME": "prepositions",
    "DOS_TO_INFINITIVE_USE": "infinitive_vs_gerund",
    "EDGAR_ALLAN_POE": "unmapped",
    "FAKE": "unmapped",
    "FIRST_AID_KIT": "unmapped",
    "FIRST_PERSON_SHOOTER": "unmapped",
    "FLASHPOINT": "unmapped",
    "FOLLOW_A_COURSE": "idioms",
    "FRISBEE": "unmapped",
    "GESTURE_OF_GREETING": "idioms",
    "GRANITE": "unmapped",
    "HAIRS": "count_uncount",
    "HANDOVER": "typos",
    "HAST_O": "typos",
    "HELL": "unmapped",
    "HELLOS": "unmapped",
    "IF_YOU_ANY": "sentence_structure",
    "IS_CAUSE_BY": "passive_voice",
    "IS_CONTAINED_OF": "passive_voice",
    "IS_RENOWN_FOR": "confusables.word_choice",
    "IT_APOS_A": "punctuation.basic",
    "IT_S_BRITNEY": "unmapped",
    "IT_TIME_TO": "sentence_structure",
    "I_A": "typos",
    "I_E": "unmapped",
    "I_EM": "typos",
    "I_IF": "typos",
    "I_ILL": "typos",
    "I_IN": "typos",
    "I_IS": "typos",
    "I_IT": "typos",
    "I_MA": "typos",
    "JJS_OF_ALL_OTHER": "comparatives_superlatives",
    "KIND_OF_A": "articles.a_an",
    "LADIES_AND_GENTLEMAN": "idioms",
    "LEARN_NNNNS_ON_DO": "verb_form",
    "LET_ME_TROUGH": "typos",
    "LICENCE_LICENSE_NOUN_PLURAL": "confusables.word_choice",
    "LICENCE_LICENSE_NOUN_SINGULAR": "confusables.word_choice",
    "LIES_THEY_SAY": "verb_form",
    "LISTEN_TO_MOVIES": "prepositions.verb",
    "LIVE_FROM_OFF": "prepositions",
    "LIVE_IN_ON_PLANET": "prepositions",
    "LOOK_FORWARD_NOT_FOLLOWED_BY_TO": "prepositions.verb",
    "LOOK_FORWARD_TO": "prepositions.verb",
    "MANEGE": "typos",
    "MATERIEL": "typos",
    "MAY_MANY_MY": "confusables.word_choice",
    "MEAN_FOR_TO": "infinitive_vs_gerund",
    "MINUETS": "typos",
    "MONTH_OF_XXXX": "punctuation.basic",
    "NAIL_ON_THE_HEAD": "idioms",
    "NAMED_IT_AS": "idioms",
    "NICE_TOO_ME": "prepositions",
    "NOTHING_SUCH_AS": "idioms",
    "NOT_SURE_IT_WORKS": "sentence_structure",
    "NOT_US1": "unmapped",
    "OPEN_TO_PAGE": "prepositions",
    "OR_WAY_IT": "sentence_structure",
    "PHOTO_WITH_HIS_CAT": "unmapped",
    "PLEASER": "typos",
    "PM_IN_THE_EVENING": "punctuation.basic",
    "POPULAR_AMONG_WITH": "prepositions.adjective",
    "PRE_AND_POST_NN": "punctuation.hyphenation",
    "PUT_FOURTH_THEAAN": "typos",
    "QUALITY_PARTITIVE_SINGULAR": "count_uncount",
    "RED_NOSED_REINDEER": "unmapped",
    "REP_THANK_YOU_FOR": "style",
    "ROAD_TO_MARKET": "idioms",
    "ROUND_A_BOUTS": "typos",
    "R_SYMBOL": "punctuation.basic",
    "SENSE_OF_FALSE_HOPEPRIVACYSECURITY": "unmapped",
    "SENTENCE_END_CONTRACT": "punctuation.basic",
    "SHOULD_BE_DO": "modal_verbs",
    "SI_UNITS_OF_MEASUREMENT_ADJECTIVES_AME": "punctuation.basic",
    "FAIR_SURE": "idioms",
    # ---- Brand / proper-noun residuals (stay unmapped) ------------------
    "SPACEX": "unmapped",
    "SPIDERMAN": "unmapped",
    "T_BONE": "unmapped",
    "T_REX": "unmapped",
    "TWITTER": "unmapped",
    "TWITTER_X": "unmapped",
    "VITAMIN_C": "unmapped",
    "VON_DER_LEYEN": "unmapped",
    "WECHAT": "unmapped",
    "WENDYS": "unmapped",
    "WERNHER_VON_BRAUN": "unmapped",
    "WIEN": "unmapped",
    "WIFI": "unmapped",
    "WORDPRESS": "unmapped",
    "YOUTUBE": "unmapped",
    "Z_WAVE": "unmapped",
    "TURKEY": "unmapped",
    "TRADEMARK": "unmapped",
    "TOMFOOLERY": "unmapped",
    "TR": "unmapped",
    "UR": "unmapped",
    "TIS": "unmapped",
    "TAT": "unmapped",
    "VERSE": "unmapped",
    "WHETHER": "unmapped",
    "TOILET": "unmapped",
    "TESTES": "unmapped",
    "SUPPER": "unmapped",
    "STAID": "unmapped",
    "WOLD": "typos",
    # ---- Past-tense / verb form errors ----------------------------------
    "SPLITTED": "verb_form",
    "SPOKED": "verb_form",
    "STRIKED": "verb_form",
    "WROTE": "verb_form",
    "WOLFS": "verb_form",
    "WENT": "verb_form",
    "UNDOS": "verb_form",
    # ---- Possessive contractions / common typos -------------------------
    "WHOS": "typos",
    "YOURE": "typos",
    "YOUR": "typos",
    "YOUR_YOU_2": "pronouns.possessive",
    "YOUR_YOU_RE": "pronouns.possessive",
    "WED_WE_D": "typos",
    "WERE_WE_RE": "typos",
    "WAS_WANDERING_IF": "typos",
    "WANNA": "typos",
    "WHATCHA": "typos",
    "WHAT_SO_EVER": "typos",
    "WAN_T": "typos",
    "WASENT": "typos",
    "WASEN_T": "typos",
    "WON_T_TO": "typos",
    "WOUN_T": "typos",
    "Y_ALL": "typos",
    "YOU_R": "typos",
    "YOU_RE_AREN_T": "typos",
    "WE_LL_WELL": "typos",
    "U_RE": "typos",
    "U_TURN": "typos",
    "VE_GO_TO": "typos",
    "TRYNA": "typos",
    "T_HE": "typos",
    "T_O": "typos",
    "W_HAT": "typos",
    "THEE": "unmapped",
    "THEIR_S": "pronouns.possessive",
    "THERETO": "unmapped",
    # ---- Idioms residuals -----------------------------------------------
    "SLIGHT_OF_HAND": "idioms",
    "SOON_OR_LATER": "idioms",
    "SPARE_OF_THE_MOMENT": "idioms",
    "STARS_AND_STEPS": "idioms",
    "STATE_OF_ART": "idioms",
    "STATE-OF-THE-ART": "punctuation.hyphenation",
    "STATE_OF_THE_ART": "idioms",
    "STATE_OF_THE_UNION": "unmapped",
    "STATUE_OF_LIMITATIONS": "idioms",
    "STOCK_AND_TRADE": "idioms",
    "STRIKE_A_CORD": "idioms",
    "TAKE_A_BATH": "idioms",
    "TAKE_A_LOOK": "idioms",
    "TAKE_INTO_ACCOUNT": "idioms",
    "TAKE_IT_PERSONAL": "idioms",
    "TAKE_THE_REIGNS": "idioms",
    "TAKING_CASE_OF_IT": "idioms",
    "TAKING_INTO_CONSIDERATION": "idioms",
    "THANKS_FOR_CONTACTING_ME": "idioms",
    "THANKS_FOR_LET_YOU_KNOW": "idioms",
    "THANKS_IN_ADVANCED": "idioms",
    "THANK_IN_ADVANCE": "idioms",
    "THAT_BEING_SAID": "style",
    "THAT_SOUND_GREAT": "subject_verb_agreement",
    "THAT_S_WHAT_S": "punctuation.basic",
    "THAT_S_YOU_RE": "confusables.word_choice",
    "THAT_VERY_COOL": "sentence_structure",
    "THE_ADD_ON": "punctuation.hyphenation",
    "THE_APO_RE": "punctuation.basic",
    "THE_BEST_WAY": "articles.definite_indefinite",
    "THE_HOT_DOG": "articles.definite_indefinite",
    "THE_LATER_LATTER": "confusables.word_choice",
    "THE_ONLY_ON": "typos",
    "THE_PROOF_IS_IN_THE_PUDDING": "idioms",
    "THE_SAME_AS": "comparatives_superlatives",
    "THE_THIRD_PARTY": "articles.definite_indefinite",
    "THE_WORSE_OF": "comparatives_superlatives",
    "THINK_YOU_A": "typos",
    "THIS_IS_HAVE": "verb_form",
    "THIS_PLURAL_OF": "subject_verb_agreement",
    "THIS_TWO_MEN": "determiners_quantifiers",
    "TIP_AND_TRICK": "idioms",
    "TONGUE_AND_CHEEK": "idioms",
    "TOO_CARDINAL_NUMBER": "punctuation.basic",
    "TOO_TO_EITHER": "sentence_structure",
    "TOW_THE_LINE": "idioms",
    "TO_ALL_INTENTS_AND_PURPOSES": "idioms",
    "TO_FOUND_FIND_A_CURE": "verb_form",
    "TO_FRESH_UP": "idioms",
    "TO_JJR_THAN": "comparatives_superlatives",
    "TO_NON_BASE": "verb_form",
    "TO_ON_A_TRIP": "prepositions",
    "TO_THE_MANOR_BORN": "idioms",
    "TO_WHO_IT_MAY_CONCERN": "pronouns.case",
    "TO__THEN_BY": "prepositions",
    "TRITE_AND_TRUE": "idioms",
    "TRUE_TO_WORD": "idioms",
    "TURN_IT_OF": "typos",
    "TWELFTH_OF_NEVER": "idioms",
    "TWICE_AS_COLD": "comparatives_superlatives",
    "VB_A_WHILE": "verb_form",
    "VERB_HERE_SINCE": "verb_tense",
    "VERY_SMALL_TINY": "style",
    "WAITING_MY_PATIENT": "prepositions.verb",
    "WAITING_MY_PATIENT_FINISH": "verb_form",
    "WANTED_TO_RE_SENT": "verb_form",
    "WANT_THAT_I": "infinitive_vs_gerund",
    "WAS_THERE_MANY": "subject_verb_agreement",
    "WET_YOUR_APPETITE": "idioms",
    "WHAT_ARE_TALKING_ABOUT": "sentence_structure",
    "WHAT_DO_THAT": "subject_verb_agreement",
    "WHAT_IS_REASON": "articles.definite_indefinite",
    "WHAT_IS_YOU": "subject_verb_agreement",
    "WHAT_IT_HAPPENING": "subject_verb_agreement",
    "WHAT_IT_THE": "sentence_structure",
    "WHAT_WE_CALL_2": "sentence_structure",
    "WHERE_YOU_FROM": "questions",
    "WHIM_AND_A_PRAYER": "idioms",
    "WILL_BASED_ON": "modal_verbs",
    "WILL_LIKE_TO": "modal_verbs",
    "WITCH_IS_WRONG": "confusables.word_choice",
    "WORD_CONTAINS_UNDERSCORE": "punctuation.basic",
    "WORK_AS_A_CHARM": "idioms",
    "WORK_LIFE_BALANCE": "punctuation.hyphenation",
    "WORSE-CASE_SCENARIO": "idioms",
    "WORST_COMES_TO_WORST": "idioms",
    "WRITE_IN_MY_OWN_PAGE": "prepositions",
    "WRB_THERE_THEY_RE": "confusables.word_choice",
    "YEAR_20001": "punctuation.basic",
    "YEAR_END_AND_YEAR_OUT": "idioms",
    "YEAR_OLD_PLURAL": "punctuation.hyphenation",
    "ZERO-SUM_GAIN": "idioms",
    "ZIP_CODE_POSTCODE": "unmapped",
    # ---- More structural / SVA / quantifier rules ----------------------
    "SUBJECT_VERB_AGREEMENT_PLURAL": "subject_verb_agreement",
    "SV_AGREEMENT_CLAUSES_PLURAL": "subject_verb_agreement",
    "SV_AGREEMENT_CLAUSES_SINGULAR": "subject_verb_agreement",
    "SUPERFLUOUS_OXFORD_ADJACENT": "punctuation.basic",
    "SUPERIOR_THAN_TO": "comparatives_superlatives",
    "TAG_QUESTIONS_2": "questions",
    "TAKEAWAY": "typos",
    "TALK_NO_PREP": "prepositions.verb",
    "TATTLE-TAIL": "typos",
    "TELL_X_TO_DO": "infinitive_vs_gerund",
    "OF_ALL_PLURAL": "determiners_quantifiers",
    "OF_ALL_TIMES": "determiners_quantifiers",
    "OLD_WISE_TAILTALE": "idioms",
    "ON_FIRST_GLANCE": "idioms",
    "ON_IN_A_MEETING": "prepositions",
    "ON_IN_CHARGE_OF": "prepositions",
    "ON_OF_THE": "prepositions",
    "ON_THE_NOVEL": "prepositions",
    "ON_THE_SHELF": "prepositions",
    "O_CLOCK": "punctuation.basic",
    "O_CONNOR": "unmapped",
    "O_TO": "typos",
    "NOT_US1": "unmapped",
    "NOW_ARE_THE_TIME": "subject_verb_agreement",
    "NO_PROBLEM_ET_AL": "idioms",
    "QB_EN_OTHER": "unmapped",
    "QB_NEW_EN_OTHER": "unmapped",
    "QB_EN_SWAP": "unmapped",
    # ---- Some-time / one-word vs two-word -------------------------------
    "SOMETIME_SOME_TIME": "typos",
    "SOME_TIMES_TIME": "typos",
    "SOME_TIME_SOMETIMES": "typos",
    "SON_T": "typos",
    "SPACE_BETWEEN_NUMBER_AND_WORD": "punctuation.basic",
    "STEPS_TO_DO": "infinitive_vs_gerund",
    "STOP_HIM_OF_FROM": "prepositions.verb",
    "STRESS_OUT_FOR_OVER": "prepositions.verb",
    "UNKNOWN_HAVE_ITS": "verb_form",
    "UP_TO_DATA": "idioms",
    "US_ONE_ENTITY": "subject_verb_agreement",
    "V_SHAPED": "punctuation.hyphenation",
}


# ---------------------------------------------------------------------------
# Priority-ordered regex heuristics. First match wins.
# Format: (compiled_pattern, target_type, reason_label)
# ---------------------------------------------------------------------------

def _compile(patterns: List[Tuple[str, str, str]]) -> List[Tuple[Pattern[str], str, str]]:
    return [(re.compile(p), t, r) for p, t, r in patterns]


HEURISTICS: List[Tuple[Pattern[str], str, str]] = _compile([
    # =====================================================================
    # AI-generated rules (suffix-driven). Most specific first.
    # =====================================================================
    (r"^AI_.*REPLACEMENT_VERB_TENSE$",                    "verb_tense",              "ai_replacement_verb_tense"),
    (r"^AI_.*REPLACEMENT_VERB_FORM$",                     "verb_form",               "ai_replacement_verb_form"),
    (r"^AI_.*REPLACEMENT_VERB_SVA$",                      "subject_verb_agreement",  "ai_replacement_verb_sva"),
    (r"^AI_.*REPLACEMENT_VERB_AGREEMENT$",                "subject_verb_agreement",  "ai_replacement_verb_agreement"),
    (r"^AI_.*REPLACEMENT_VERB$",                          "verb_form",               "ai_replacement_verb"),
    (r"^AI_.*REPLACEMENT_NOUN_NUMBER$",                   "count_uncount",           "ai_replacement_noun_number"),
    (r"^AI_.*REPLACEMENT_NOUN_FORM$",                     "count_uncount",           "ai_replacement_noun_form"),
    (r"^AI_.*REPLACEMENT_NOUN$",                          "count_uncount",           "ai_replacement_noun"),
    (r"^AI_.*REPLACEMENT_DETERMINER$",                    "articles.definite_indefinite", "ai_replacement_determiner"),
    (r"^AI_.*REPLACEMENT_WORD_ORDER$",                    "word_order",              "ai_replacement_word_order"),
    (r"^AI_.*REPLACEMENT_ORTHOGRAPHY(_\w+)?$",            "typos",                   "ai_replacement_orthography"),
    (r"^AI_.*REPLACEMENT_ORTH(_\w+)?$",                   "typos",                   "ai_replacement_orth"),
    (r"^AI_.*REPLACEMENT_PUNCTUATION$",                   "punctuation.basic",       "ai_replacement_punctuation"),
    (r"^AI_.*REPLACEMENT_CONFUSION$",                     "confusables.word_choice", "ai_replacement_confusion"),
    (r"^AI_.*REPLACEMENT_CONJUNCTION$",                   "sentence_structure",      "ai_replacement_conjunction"),
    (r"^AI_.*MISSING_DETERMINER(_\w+)?$",                 "articles.definite_indefinite", "ai_missing_determiner"),
    (r"^AI_.*MISSING_PUNCTUATION(_\w+)?$",                "punctuation.basic",       "ai_missing_punctuation"),
    (r"^AI_.*MISSING_ORTHOGRAPHY(_\w+)?$",                "typos",                   "ai_missing_orthography"),
    (r"^AI_.*UNNECESSARY_PUNCTUATION(_\w+)?$",            "punctuation.basic",       "ai_unnecessary_punctuation"),
    (r"^AI_.*UNNECESSARY_DETERMINER$",                    "articles.definite_indefinite", "ai_unnecessary_determiner"),
    (r"^AI_.*UNNECESSARY_NOUN$",                          "count_uncount",           "ai_unnecessary_noun"),
    (r"^AI_.*UNNECESSARY_ADPOSITION$",                    "prepositions",            "ai_unnecessary_adposition"),
    (r"^AI_.*DELETE_VERB$",                               "verb_form",               "ai_delete_verb"),
    (r"^AI_HYDRA_LEO_CP_.*$",                             "confusables.homophones",  "ai_hydra_cp"),
    (r"^AI_HYDRA_LEO_CPT_.*$",                            "confusables.homophones",  "ai_hydra_cpt"),
    (r"^AI_HYDRA_LEO_MISSING_(A|THE)$",                   "articles.definite_indefinite", "ai_hydra_missing_article"),
    (r"^AI_HYDRA_LEO_MISSING_(IN|OF|TO|AT|ON)$",          "prepositions",            "ai_hydra_missing_prep"),
    (r"^AI_HYDRA_LEO_MISSING_COMMA$",                     "punctuation.basic",       "ai_hydra_missing_comma"),
    (r"^AI_HYDRA_LEO_MISSING_IT$",                        "pronouns",                "ai_hydra_missing_it"),
    (r"^AI_HYDRA_LEO_APOSTROPHE.*$",                      "punctuation.basic",       "ai_hydra_apostrophe"),
    # AI catchall: stays unmapped (REPLACEMENT_OTHER, INSERT_OTHER, etc.)

    # =====================================================================
    # Hyphenation
    # =====================================================================
    (r"^.*_HYPHEN(ATED)?$",                               "punctuation.hyphenation", "suffix_hyphen"),
    (r"^HYPHEN_.*$",                                      "punctuation.hyphenation", "prefix_hyphen"),
    (r"^.*HYPHENATION.*$",                                "punctuation.hyphenation", "contains_hyphenation"),
    (r"^ADVERB_OR_HYPHENATED_ADJECTIVE$",                 "punctuation.hyphenation", "explicit_hyphen_adv_adj"),

    # =====================================================================
    # Word order
    # =====================================================================
    (r"^.*WORD_ORDER.*$",                                 "word_order",              "contains_word_order"),

    # =====================================================================
    # Subject-verb agreement
    # =====================================================================
    (r"^AGREEMENT_.*$",                                   "subject_verb_agreement",  "prefix_agreement"),
    (r"^.+_AGREEMENT(_\d+)?$",                            "subject_verb_agreement",  "suffix_agreement"),
    (r"^SVA_.*$",                                         "subject_verb_agreement",  "prefix_sva"),
    (r"^.+_SVA$",                                         "subject_verb_agreement",  "suffix_sva"),

    # =====================================================================
    # Sentence structure (ahead of generic SENT/COMMA rules)
    # =====================================================================
    (r"^FRAGMENT_.*$",                                    "sentence_structure",      "fragment"),
    (r"^.*FRAGMENT.*$",                                   "sentence_structure",      "contains_fragment"),
    (r"^RUN_ON.*$",                                       "sentence_structure",      "run_on"),
    (r"^.*COMMA_SPLICE.*$",                               "sentence_structure",      "comma_splice"),
    (r"^.+_SENT_(START|END).*$",                          "sentence_structure",      "sent_start_end"),
    (r"^SENT_START_.*$",                                  "sentence_structure",      "sent_start_prefix"),

    # =====================================================================
    # Punctuation (apostrophe, comma, period, etc.)
    # =====================================================================
    (r"^APOS$",                                           "punctuation.basic",       "apos_lone"),
    (r"^APOS_.*$",                                        "punctuation.basic",       "prefix_apos"),
    (r"^APOSTROPHE_.*$",                                  "punctuation.basic",       "prefix_apostrophe"),
    (r"^.+_APOSTROPHE.*$",                                "punctuation.basic",       "contains_apostrophe"),
    (r"^.+_PUNCTUATION$",                                 "punctuation.basic",       "suffix_punctuation"),
    (r"^PUNCTUATION_.*$",                                 "punctuation.basic",       "prefix_punctuation"),
    (r"^COMMA_.*$",                                       "punctuation.basic",       "prefix_comma"),
    (r"^.+_COMMA$",                                       "punctuation.basic",       "suffix_comma"),
    (r"^DASH_.*$",                                        "punctuation.basic",       "dash"),
    (r"^SEMICOLON_.*$",                                   "punctuation.basic",       "semicolon"),
    (r"^COLON_.*$",                                       "punctuation.basic",       "colon"),

    # =====================================================================
    # Articles & determiners
    # =====================================================================
    (r"^A_VS_AN.*$",                                      "articles.a_an",           "a_vs_an"),
    (r"^AN_VS_A.*$",                                      "articles.a_an",           "an_vs_a"),
    (r"^EN_A_VS_AN.*$",                                   "articles.a_an",           "en_a_vs_an"),
    (r"^ARTICLE_.*$",                                     "articles.definite_indefinite", "prefix_article"),
    (r"^.+_ARTICLE_MISSING$",                             "articles.definite_indefinite", "suffix_article_missing"),
    (r"^MISSING_ARTICLE.*$",                              "articles.definite_indefinite", "missing_article"),
    (r"^MISSING_DETERMINER.*$",                           "articles.definite_indefinite", "missing_determiner"),
    (r"^MISSING_THE.*$",                                  "articles.definite_indefinite", "missing_the"),
    (r"^MISSING_A_AN$",                                   "articles.a_an",           "missing_a_an"),
    (r"^.*UNCOUNTABLE.*$",                                "count_uncount",           "uncountable"),

    # =====================================================================
    # Pronouns
    # =====================================================================
    (r"^.*MYSELF.*$",                                     "pronouns.reflexive",      "myself"),
    (r"^.*HIMSELF.*$",                                    "pronouns.reflexive",      "himself"),
    (r"^.*HERSELF.*$",                                    "pronouns.reflexive",      "herself"),
    (r"^.*OURSELVES.*$",                                  "pronouns.reflexive",      "ourselves"),
    (r"^.*YOURSELF.*$",                                   "pronouns.reflexive",      "yourself"),
    (r"^.*THEMSELVES.*$",                                 "pronouns.reflexive",      "themselves"),
    (r"^.+_POSSESSIVE.*$",                                "pronouns.possessive",     "suffix_possessive"),
    (r"^POSSESSIVE_.*$",                                  "pronouns.possessive",     "prefix_possessive"),
    (r"^PRP_.*$",                                         "pronouns",                "prefix_prp"),
    (r"^.+_PRP$",                                         "pronouns",                "suffix_prp"),
    (r"^.+_PRP_.*$",                                      "pronouns",                "infix_prp"),
    (r"^CONFUSION_OF_(ME|I|HE|HIM|SHE|HER|US|WE|YOU|THEY|THEM).*$", "pronouns.case", "pronoun_case_confusion"),

    # =====================================================================
    # Verb tense / form / modal / passive / inf-vs-gerund
    # =====================================================================
    (r"^.*PASSIVE.*$",                                    "passive_voice",           "passive"),
    (r"^.*MODAL.*$",                                      "modal_verbs",             "modal"),
    (r"^MD_.*$",                                          "modal_verbs",             "prefix_md"),
    (r"^.+_MD_VB$",                                       "modal_verbs",             "md_vb"),
    (r"^MUST_(VB|VBP|VBN).*$",                            "modal_verbs",             "must_vb"),
    (r"^SHOULD_(VB|VBP|VBN|HAVE).*$",                     "modal_verbs",             "should_vb"),
    (r"^WOULD_(VB|VBP|VBN|HAVE).*$",                      "modal_verbs",             "would_vb"),
    (r"^COULD_(VB|VBP|VBN|HAVE).*$",                      "modal_verbs",             "could_vb"),
    (r"^.*INFINITIVE.*$",                                 "infinitive_vs_gerund",    "infinitive"),
    (r"^.*GERUND.*$",                                     "infinitive_vs_gerund",    "gerund"),
    (r"^TO_VBG_.*$",                                      "infinitive_vs_gerund",    "to_vbg"),
    (r"^.+_TO_VBG$",                                      "infinitive_vs_gerund",    "suffix_to_vbg"),
    (r"^.*_TENSE$",                                       "verb_tense",              "suffix_tense"),
    (r"^.*_VERB_TENSE$",                                  "verb_tense",              "verb_tense"),
    (r"^.*_VERB_FORM$",                                   "verb_form",               "verb_form"),
    (r"^.*BASE_FORM.*$",                                  "verb_form",               "base_form"),
    (r"^AUXILIARY_.*$",                                   "auxiliary_verbs",         "auxiliary_prefix"),
    (r"^.+_AUXILIARY.*$",                                 "auxiliary_verbs",         "auxiliary_infix"),

    # =====================================================================
    # Negation
    # =====================================================================
    (r"^.*DOUBLE_NEG.*$",                                 "negation.double_negation","double_negation"),
    (r"^DOUBLE_NEGATION.*$",                              "negation.double_negation","double_negation_prefix"),
    (r"^.*_NEGATION.*$",                                  "negation",                "negation"),
    (r"^NEGATION_.*$",                                    "negation",                "negation_prefix"),

    # =====================================================================
    # Comparatives / superlatives
    # =====================================================================
    (r"^.*COMPARATIVE.*$",                                "comparatives_superlatives","comparative"),
    (r"^.*SUPERLATIVE.*$",                                "comparatives_superlatives","superlative"),
    (r"^MORE_BETTER.*$",                                  "comparatives_superlatives","more_better"),
    (r"^MORE_THAN_.*$",                                   "comparatives_superlatives","more_than"),
    (r"^MOST_BEST.*$",                                    "comparatives_superlatives","most_best"),

    # =====================================================================
    # Conditionals
    # =====================================================================
    (r"^.*CONDITIONAL.*$",                                "conditionals",            "conditional"),
    (r"^IF_(I|HE|SHE|WE|THEY|YOU|IT)_(VB|VBP|VBD|VBN|WAS|WERE|WOULD).*$", "conditionals", "if_subject_verb"),
    (r"^WOULD_OF$",                                       "conditionals",            "would_of"),
    (r"^WOULD_HAVE_.*$",                                  "conditionals",            "would_have"),
    (r"^.*WOULD_HAVE_VB$",                                "conditionals",            "would_have_vb"),

    # =====================================================================
    # Questions
    # =====================================================================
    (r"^.*QUESTION_MARK.*$",                              "punctuation.basic",       "question_mark"),
    (r"^.*_QUESTION$",                                    "questions",               "suffix_question"),
    (r"^QUESTION_.*$",                                    "questions",               "prefix_question"),

    # =====================================================================
    # Relative clauses
    # =====================================================================
    (r"^WHICH_THAT$",                                     "relative_clauses",        "which_that"),
    (r"^WHO_THAT$",                                       "relative_clauses",        "who_that"),
    (r"^.*RELATIVE_CLAUSE.*$",                            "relative_clauses",        "relative_clause"),

    # =====================================================================
    # Adjective vs adverb
    # =====================================================================
    (r"^.*ADJECTIVE_ADVERB.*$",                           "adjectives",              "adj_adv"),
    (r"^ADV_VS_ADJ.*$",                                   "adverbs",                 "adv_vs_adj"),
    (r"^ADJ_VS_ADV.*$",                                   "adjectives",              "adj_vs_adv"),

    # =====================================================================
    # Prepositions (generic)
    # =====================================================================
    (r"^PREPOSITION_.*$",                                 "prepositions",            "prefix_preposition"),
    (r"^.+_PREPOSITION$",                                 "prepositions",            "suffix_preposition"),

    # =====================================================================
    # CONFUSION_* family (LT explicit confusable rules)
    # =====================================================================
    (r"^CONFUSION_OF_(ME|I|HE|HIM|SHE|HER|US|WE|YOU|THEY|THEM)_.*$", "pronouns.case", "confusion_pronoun_case"),
    (r"^CONFUSION_.*$",                                   "confusables.word_choice", "confusion_prefix"),
    (r"^.*_CONFUSION$",                                   "confusables.word_choice", "suffix_confusion"),

    # =====================================================================
    # Compound adjectives & compounds
    # =====================================================================
    (r"^CA_[A-Z_]+$",                                     "punctuation.hyphenation", "ca_compound_adjective"),
    (r"^.+_COMPOUND$",                                    "punctuation.hyphenation", "suffix_compound"),
    (r"^.+_COMPOUNDS$",                                   "punctuation.hyphenation", "suffix_compounds"),
    (r"^COMPOUND_.*$",                                    "punctuation.hyphenation", "prefix_compound"),

    # =====================================================================
    # Capitalization / orthography
    # =====================================================================
    (r"^.*CAPITALIZATION.*$",                             "punctuation.basic",       "capitalization"),
    (r"^LOWERCASE_.*$",                                   "punctuation.basic",       "lowercase"),
    (r"^.+_CASING$",                                      "punctuation.basic",       "casing"),
    (r"^.*UPPERCASE.*$",                                  "punctuation.basic",       "uppercase"),

    # =====================================================================
    # Missing / Unnecessary punctuation
    # =====================================================================
    (r"^MISSING_COMMA.*$",                                "punctuation.basic",       "missing_comma"),
    (r"^MISSING_PERIOD.*$",                               "punctuation.basic",       "missing_period"),
    (r"^MISSING_TO_BEFORE_A_VERB$",                       "infinitive_vs_gerund",    "missing_to_before_verb"),
    (r"^MISSING_TO_BETWEEN_BE_AND_VB$",                   "infinitive_vs_gerund",    "missing_to_between"),

    # =====================================================================
    # Date / multitoken-spelling specifics
    # =====================================================================
    (r"^EN_DATE_.*$",                                     "typos",                   "en_date"),
    (r"^EN_MULTITOKEN_SPELLING_.*$",                      "typos",                   "en_multitoken"),
    (r"^EN_UNPAIRED.*$",                                  "punctuation.basic",       "en_unpaired"),

    # =====================================================================
    # Determiner tag rules
    # =====================================================================
    (r"^DT_.*$",                                          "articles.definite_indefinite", "prefix_dt"),
    (r"^.+_DT$",                                          "articles.definite_indefinite", "suffix_dt"),

    # =====================================================================
    # Verb-tag suffixes (verb forms / tense)
    # =====================================================================
    (r"^.+_VBN$",                                         "verb_form",               "suffix_vbn"),
    (r"^.+_VBD$",                                         "verb_tense",              "suffix_vbd"),
    (r"^.+_VBZ$",                                         "subject_verb_agreement",  "suffix_vbz"),
    (r"^.+_VBP$",                                         "verb_form",               "suffix_vbp"),
    (r"^.+_VB$",                                          "verb_form",               "suffix_vb"),
    (r"^.+_VBG$",                                         "infinitive_vs_gerund",    "suffix_vbg"),

    # =====================================================================
    # Auxiliaries: do/does/did and contractions
    # =====================================================================
    (r"^DO_(NOTHING|NO_VB|TO|ABLE|ANYONE|ARTS|A_PARTY|HAVE_A_MEETING|HE_VERB|IT_SOMETIME|TO_THE_FACT_THAT|TO_THE_LACK_OF|YOU_FASCINATED|YOU_WHAT)$",
        "auxiliary_verbs", "do_aux"),
    (r"^DID_(A_MISTAKE|BASEFORM|FOUND_AMBIGUOUS|PAST|YOU_HAVE_VBN).*$", "verb_form", "did_aux"),
    (r"^DON_T_AREN_T$",                                   "auxiliary_verbs",         "dont_arent"),
    (r"^DO(N(N)?_T|NS_T|NT_T|N_T_VBZ|NTCHA|NT_WHAT|NT_WILL|NT_NEEDS)$", "typos",     "dont_typo"),
    (r"^IF_WOULD_HAVE_VBN$",                              "conditionals",            "if_would_have"),

    # =====================================================================
    # Misc structural / agreement helpers
    # =====================================================================
    (r"^IS_(AND_ARE|VBZ|WAS|WERE|RB_BE|EVEN_WORST|US|OWN|POSSIBLE_TO|PLEASURE_TO|THERE_ANY_NNS)$",
        "subject_verb_agreement", "is_agr"),
    (r"^HOW_(IS_ARE|DOES_THIS_CHANGES|YOU_DOING)$",        "subject_verb_agreement", "how_agr"),
    (r"^IT_IS(_2|_NO|_SURE)?$",                           "sentence_structure",      "it_is"),
    (r"^MISSING_VERB_AFTER_WHAT$",                        "sentence_structure",      "missing_verb"),

    # =====================================================================
    # FOR_*, IN_*, AT_* prepositional patterns (after specific overrides)
    # =====================================================================
    (r"^FOR_VB$",                                         "prepositions.verb",       "for_vb"),
    (r"^FOR_(WHILE|AWHILE|ALONG_TIME|TIME_TO_TIME|SOMETIME_FOR_SOME_TIME|EVER_CA|EVER_GB)$",
        "typos", "for_typo"),
    (r"^FOR_(INCONVENIENCE|MY|ITS_NN|NOUN_SAKE|WHATEVER_REASONS|FRO|SELL)$",
        "prepositions", "for_prep"),
    (r"^IN_(LAWS|TACT|TITLED|ANYWAY|NOWADAYS|MASSE|STEAD_OF|TERM_OF_PHRASE|THE_MEAN_TIME_PHRASE)$",
        "typos", "in_typo"),
    (r"^IN_(WEEKDAY|WEBSITE|FACEBOOK|TWITTER|WINDOWS|CHINA|QUEENS|SHANGHAI|LONG_BEACH|LONG_ISLAND|YOU_RE_NN)$",
        "prepositions.place", "in_place"),
    (r"^IN_(CHARGE_FOR|CHARGE_OF_FROM|EDITION_TO|TO_INTO|TO_VBD|VEIN|PRINCIPAL|FRONT_OF|VBZ_THEYRE_NN|WHO|X_ORDER|ALONG_TIME|THIS_REGARDS|THIS_MOMENT|A_HARRY|A_ISLAND|A_TROUBLE|A_X_MANNER|LOVED_WITH|SANE|ON_A_SECRET_MISSION|ON_A_TEAM|ON_A_TRIP|ON_BIRTHDAY|ON_THE_FOOT|ON_THE_TEAM|ON_VACATION|AT_A_PARTY|1990s)$",
        "prepositions", "in_prep"),

    # =====================================================================
    # QB_EN_* family (LanguageTool internal naming)
    # =====================================================================
    (r"^QB_EN_DECAPITALIZE.*$",                           "punctuation.basic",       "qb_decapitalize"),
    (r"^QB_EN_CAPITALIZE.*$",                             "punctuation.basic",       "qb_capitalize"),
    (r"^QB_NEW_EN_DECAPITALIZE.*$",                       "punctuation.basic",       "qb_new_decapitalize"),
    (r"^QB_EN_DELETE_PERIOD.*$",                          "punctuation.basic",       "qb_delete_period"),
    (r"^QB_EN_INSERT_COMMA.*$",                           "punctuation.basic",       "qb_insert_comma"),
    (r"^QB_EN_INSERT_PERIOD.*$",                          "punctuation.basic",       "qb_insert_period"),
    (r"^QB_EN_INSERT_NNP.*$",                             "punctuation.basic",       "qb_insert_nnp"),
    (r"^QB_EN_OXFORD.*$",                                 "punctuation.basic",       "qb_oxford"),
    (r"^QB_EN_SWAP_PERIOD.*$",                            "punctuation.basic",       "qb_swap_period"),
    (r"^QB_EN_SWAP_POS.*$",                               "punctuation.basic",       "qb_swap_pos"),
    (r"^QB_EN_SWAP_OPEN_QUOTE.*$",                        "punctuation.basic",       "qb_swap_quote"),
    (r"^QB_EN_DELETE_DT.*$",                              "articles.definite_indefinite", "qb_delete_dt"),
    (r"^QB_EN_INSERT_DT.*$",                              "articles.definite_indefinite", "qb_insert_dt"),
    (r"^QB_EN_DELETE_IN.*$",                              "prepositions",            "qb_delete_in"),
    (r"^QB_EN_DELETE_UNDERSCORESP.*$",                    "punctuation.basic",       "qb_delete_underscore"),
    (r"^QB_EN_DELETE_NFP.*$",                             "punctuation.basic",       "qb_delete_nfp"),
    (r"^QB_EN_SWAP_DT.*$",                                "articles.definite_indefinite", "qb_swap_dt"),
    (r"^QB_EN_SWAP_NNS.*$",                               "count_uncount",           "qb_swap_nns"),
    (r"^QB_EN_SWAP_NN_FOR_RB.*$",                         "adjectives",              "qb_swap_nn_rb"),
    (r"^QB_EN_SWAP_NN.*$",                                "count_uncount",           "qb_swap_nn"),
    (r"^QB_EN_SWAP_NNP.*$",                               "punctuation.basic",       "qb_swap_nnp"),
    (r"^QB_EN_SWAP_RB.*$",                                "adverbs",                 "qb_swap_rb"),
    (r"^QB_EN_SWAP_JJ.*$",                                "adjectives",              "qb_swap_jj"),
    (r"^QB_EN_SWAP_CC.*$",                                "sentence_structure",      "qb_swap_cc"),
    (r"^QB_EN_SWAP_CD.*$",                                "punctuation.basic",       "qb_swap_cd"),
    (r"^QB_TEST_EN_JOINED_MATCH$",                        "typos",                   "qb_joined_match"),
    (r"^QB_EN_MERGED_MATCH.*$",                           "typos",                   "qb_merged_match"),

    # =====================================================================
    # Oxford / British-American spelling
    # =====================================================================
    (r"^OXFORD_SPELLING.*$",                              "typos",                   "oxford_spelling"),
    (r"^OXFORD_NEW_CLAUSE$",                              "punctuation.basic",       "oxford_clause"),
    (r"^SERIAL_COMMA_(ON|OFF)$",                          "punctuation.basic",       "serial_comma"),

    # =====================================================================
    # Non-standard / numbers / orthography
    # =====================================================================
    (r"^NUMBERS_IN_WORDS$",                               "typos",                   "numbers_in_words"),
    (r"^NON_STANDARD_.*$",                                "typos",                   "non_standard"),
    (r"^NON_ENGLISH_CHARACTER_IN_A_WORD$",                "typos",                   "non_english_char"),
    (r"^NON3PRS_VERB$",                                   "subject_verb_agreement",  "non_3rd_person"),
    (r"^NON_ACTION_CONTINUOUS$",                          "verb_form",               "non_action_cont"),
    (r"^NON_ANTI_JJ$",                                    "adjectives",              "non_anti_jj"),
    (r"^ORDINAL_NUMBER_SUFFIX$",                          "punctuation.basic",       "ordinal_suffix"),

    # =====================================================================
    # NN/NNS infixes (noun-related)
    # =====================================================================
    (r"^.+_NNS$",                                         "count_uncount",           "suffix_nns"),
    (r"^.+_NN$",                                          "articles.a_an",           "suffix_nn"),
    (r"^.+_NNP$",                                         "punctuation.basic",       "suffix_nnp"),
    (r"^.+_NN_.*$",                                       "count_uncount",           "infix_nn"),

    # =====================================================================
    # TYPO_* prefix (LT explicitly tagged typo rules)
    # =====================================================================
    (r"^TYPO_.*$",                                        "typos",                   "typo_prefix"),

    # =====================================================================
    # WH_* family (WH-question agreement / structure)
    # =====================================================================
    (r"^WH_AUX_.*$",                                      "questions",               "wh_aux"),
    (r"^WH_VERB_.*$",                                     "questions",               "wh_verb"),

    # =====================================================================
    # THERE_*, THIS_*, THAT_* sentence-structure patterns
    # =====================================================================
    (r"^THERE_(IS|RE|S|WAS|WERE)_.*$",                    "subject_verb_agreement",  "there_sva"),
    (r"^THERE_MISSING_VERB$",                             "sentence_structure",      "there_missing_verb"),
    (r"^THIS_MISSING_VERB$",                              "sentence_structure",      "this_missing_verb"),
    (r"^.+_MISSING_VERB$",                                "sentence_structure",      "missing_verb_suffix"),

    # Two-token fallback removed — too aggressive. Run subdivide_confusables.py
    # after bulk classification if new unmapped rules appear.
])


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(rule_id: str) -> Optional[Tuple[str, str]]:
    """Return (target_type, reason) or None if no heuristic matches."""
    if rule_id in EXPLICIT_OVERRIDES:
        return EXPLICIT_OVERRIDES[rule_id], "explicit_override"
    for pattern, target, reason in HEURISTICS:
        if pattern.match(rule_id):
            return target, reason
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without modifying the mapping file.",
    )
    parser.add_argument(
        "--show-residual",
        type=int,
        default=40,
        help="Number of residual unmapped rules to print (default: 40).",
    )
    args = parser.parse_args()

    if not MAPPING_PATH.exists():
        print(f"Mapping file not found: {MAPPING_PATH}", file=sys.stderr)
        return 1
    if not VOCAB_PATH.exists():
        print(f"Vocab file not found: {VOCAB_PATH}", file=sys.stderr)
        return 1

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocab = json.load(f).get("categories", [])
    valid_targets = set(vocab)

    unmapped_ids = sorted([k for k, v in mapping.items() if v == UNMAPPED])

    moves: dict[str, list[tuple[str, str]]] = defaultdict(list)
    residual: list[str] = []
    invalid: list[tuple[str, str]] = []

    for rid in unmapped_ids:
        result = classify(rid)
        if result is None:
            residual.append(rid)
            continue
        target, reason = result
        if target == UNMAPPED:
            residual.append(rid)
            continue
        if target not in valid_targets:
            invalid.append((rid, target))
            continue
        moves[target].append((rid, reason))

    moved_count = sum(len(v) for v in moves.values())

    print("=" * 72)
    print("Classification summary")
    print("=" * 72)
    print(f"Total currently unmapped:  {len(unmapped_ids)}")
    print(f"Will move to typed bucket: {moved_count}")
    print(f"Stays in unmapped:         {len(residual)}")
    print(f"Invalid target (skipped):  {len(invalid)}")
    print()

    print("Moves by target type:")
    for target in sorted(moves.keys()):
        items = moves[target]
        sample = ", ".join(rid for rid, _ in items[:3])
        print(f"  {target:36s}  +{len(items):4d}   e.g. {sample}")

    if invalid:
        print()
        print("Invalid targets (typo in HEURISTICS or vocab missing entry):")
        for rid, t in invalid[:20]:
            print(f"  {rid} -> {t!r}")

    final_dist_preview: Counter = Counter(mapping.values())
    for target, items in moves.items():
        final_dist_preview[UNMAPPED] -= len(items)
        final_dist_preview[target] += len(items)

    total = sum(final_dist_preview.values())
    print()
    print("Projected final distribution (top 10 + tails):")
    sorted_dist = sorted(final_dist_preview.items(), key=lambda x: -x[1])
    for t, c in sorted_dist[:15]:
        pct = 100.0 * c / total if total else 0.0
        print(f"  {t:36s} {c:5d}   {pct:5.1f}%")

    if args.dry_run:
        print()
        print("Dry run complete. No files modified.")
        with RESIDUAL_REPORT_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {"residual_unmapped": sorted(residual), "count": len(residual)},
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
        print(f"Residual list written to {RESIDUAL_REPORT_PATH}")
        if args.show_residual > 0 and residual:
            print()
            print(f"First {min(args.show_residual, len(residual))} residual rule IDs:")
            for rid in residual[: args.show_residual]:
                print(f"  {rid}")
        return 0

    shutil.copy(MAPPING_PATH, BACKUP_PATH)
    print(f"Backup written to {BACKUP_PATH}")

    for target, items in moves.items():
        for rid, _reason in items:
            mapping[rid] = target

    with MAPPING_PATH.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(mapping.items())), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Mapping updated: {MAPPING_PATH}")

    with RESIDUAL_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {"residual_unmapped": sorted(residual), "count": len(residual)},
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    print(f"Residual list written to {RESIDUAL_REPORT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
