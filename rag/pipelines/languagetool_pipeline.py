"""LanguageTool-based mistake detection pipeline."""
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

import requests

try:
    from language_tool_python import LanguageTool, LanguageToolPublicAPI
except ImportError:
    LanguageTool = None
    LanguageToolPublicAPI = None

from rag.embedder import Embedder
from rag.utils.assets import load_languagetool_mapping
from rag.utils.mistake_logic_vector import generate_mistake_logic_vector
from rag.utils.rule_id_normalizer import normalize_rule_id
from rag.utils.sentence_splitter import split_sentences

_lt_log = logging.getLogger(__name__)


def _lt_rule_id_raw(match: Any) -> str:
    v = getattr(match, "rule_id", None) or getattr(match, "ruleId", None)
    if v is None:
        return ""
    return str(v).strip()


def _mistake_type_and_stored_rule_id(
    rule_mapping: Dict[str, str], raw_rule_id: str
) -> Tuple[str, str]:
    """
    Resolve mistake_type from assets. Order of lookup:
    1) exact string from LanguageTool (handles keys like in_excess_of in JSON)
    2) lowercased raw (handles IN_EXCESS_OF vs in_excess_of)
    3) normalized id (uppercase, strip [N]; handles TOT_HE[1] -> TOT_HE)

    Stored rule_id is the stripped LT value so APIs show what the checker returned.
    """
    r = raw_rule_id.strip() if raw_rule_id else ""
    if not r:
        return "unlisted", ""
    normalized = normalize_rule_id(r)
    mistake_type = (
        rule_mapping.get(r)
        or rule_mapping.get(r.lower())
        or rule_mapping.get(normalized)
    )
    if mistake_type is None:
        mistake_type = "unlisted"
    return mistake_type, r


def create_language_tool() -> Optional[Any]:
    """
    Build a LanguageTool instance: remote server (LANGUAGETOOL_URL) first, else public API.
    Returns None if language_tool_python is unavailable or both paths fail.
    """
    if LanguageTool is None and LanguageToolPublicAPI is None:
        return None
    lt_url = os.environ.get("LANGUAGETOOL_URL", "").strip()
    if lt_url and LanguageTool is not None:
        try:
            return LanguageTool("en-US", remote_server=lt_url.rstrip("/"))
        except (requests.RequestException, ConnectionError, TimeoutError, OSError) as e:
            _lt_log.info(
                "LanguageTool remote server unavailable (%s), trying public API",
                e,
            )
        except Exception as e:
            _lt_log.warning(
                "LanguageTool remote init failed: %s",
                e,
                exc_info=True,
            )
    if LanguageToolPublicAPI is not None:
        try:
            return LanguageToolPublicAPI("en-US")
        except (requests.RequestException, ConnectionError, TimeoutError, OSError) as e:
            _lt_log.warning("LanguageTool public API unavailable: %s", e)
        except Exception as e:
            _lt_log.warning(
                "LanguageTool public API init failed: %s",
                e,
                exc_info=True,
            )
    return None


def process_text(
    text: str,
    user_id: str,
    user_text_id: str,
    session_id: Optional[str],
    detected_at: datetime,
    embedder: Embedder,
    source: str = "raw_text",
    lt_tool: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Process text through LanguageTool and generate mistake events.
    
    Args:
        text: Input text to check
        user_id: User identifier
        user_text_id: ID linking back to source text/attempt (user_text_id or exercise_attempt_id)
        session_id: Optional session identifier (None for exercise attempts)
        detected_at: Wall-clock instant for this submission (shared across all events from this text)
        embedder: Embedder instance for context vectors
        source: "raw_text" or "exercise_attempt"
        lt_tool: Optional LanguageTool instance (for testing)
    
    Returns:
        List of mistake event dicts, one per detected rule match.
        All events share the same user_text_id, session_id, and detected_at.
    """
    if LanguageTool is None and LanguageToolPublicAPI is None and lt_tool is None:
        raise ImportError("language_tool_python not available")

    if lt_tool is None:
        lt_tool = create_language_tool()
    if lt_tool is None:
        raise ImportError("LanguageTool is unavailable. Check LANGUAGETOOL_URL or internet connection.")
    
    rule_mapping = load_languagetool_mapping()
    sentences = split_sentences(text)
    events: List[Dict[str, Any]] = []
    
    weight = 0.5 if source == "exercise_attempt" else 1.0
    
    for sentence in sentences:
        matches = lt_tool.check(sentence)
        if not matches:
            continue
        
        # Deduplicate by ruleId per sentence
        seen_rule_ids: set = set()
        
        for match in matches:
            raw_id = _lt_rule_id_raw(match)
            norm_for_dedup = normalize_rule_id(raw_id)
            dedup_key = norm_for_dedup if norm_for_dedup else raw_id
            if not dedup_key or dedup_key in seen_rule_ids:
                continue
            seen_rule_ids.add(dedup_key)

            mistake_type, rule_id = _mistake_type_and_stored_rule_id(rule_mapping, raw_id)
            if not rule_id:
                continue
            mistake_id = str(uuid.uuid4())
            
            # Rule message from LanguageTool (for lesson context). Only for non-other/non-style.
            rule_message = ""
            if mistake_type not in ("other", "style"):
                rule_message = str(getattr(match, "message", "") or "")
            
            # Generate vectors (skip context_vector for exercise_attempt or other/style - no example_point)
            need_context = (
                source != "exercise_attempt"
                and mistake_type not in ("other", "style")
            )
            context_vector = embedder.embed_single(sentence) if need_context else [0.0] * 384
            if need_context and len(context_vector) != 384:
                raise ValueError(f"Expected 384-dim context vector, got {len(context_vector)}")
            
            mistake_logic_vector = generate_mistake_logic_vector(mistake_type)
            if len(mistake_logic_vector) != 64:
                raise ValueError(f"Expected 64-dim mistake_logic vector, got {len(mistake_logic_vector)}")
            
            event = {
                "mistake_id": mistake_id,
                "user_id": user_id,
                "user_text_id": user_text_id,  # Links back to source text/attempt
                "session_id": session_id or "",
                "rule_id": rule_id,
                "mistake_type": mistake_type,
                "rule_message": rule_message,  # LanguageTool message for lesson context
                "text": sentence,
                "source": source,
                "weight": weight,
                "detected_at": detected_at.isoformat(),
                "mistake_logic_vector": mistake_logic_vector,
                "context_vector": context_vector,
                "extra": {},
            }
            events.append(event)
    
    return events
