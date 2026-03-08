"""LanguageTool-based mistake detection pipeline."""
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import os

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


def process_text(
    text: str,
    user_id: str,
    user_text_id: str,
    session_id: Optional[str],
    timestamp: datetime,
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
        timestamp: Timestamp of text submission (shared across all events from this text)
        embedder: Embedder instance for context vectors
        source: "raw_text" or "exercise_attempt"
        lt_tool: Optional LanguageTool instance (for testing)
    
    Returns:
        List of mistake event dicts, one per detected rule match.
        All events share the same user_text_id, session_id, and timestamp.
    """
    if LanguageTool is None and LanguageToolPublicAPI is None and lt_tool is None:
        raise ImportError("language_tool_python not available")

    if lt_tool is None:
        lt_url = os.environ.get("LANGUAGETOOL_URL", "").strip()
        if lt_url and LanguageTool is not None:
            try:
                lt_tool = LanguageTool("en-US", remote_server=lt_url.rstrip("/"))
            except Exception:
                lt_tool = None
        if lt_tool is None and LanguageToolPublicAPI is not None:
            try:
                lt_tool = LanguageToolPublicAPI("en-US")
            except Exception:
                lt_tool = None
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
            rule_id = normalize_rule_id(getattr(match, "rule_id", None) or getattr(match, "ruleId", None))
            if not rule_id or rule_id in seen_rule_ids:
                continue
            seen_rule_ids.add(rule_id)
            
            mistake_type = rule_mapping.get(rule_id, "unlisted")
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
                "timestamp": timestamp.isoformat(),  # Shared timestamp from source text/attempt
                "mistake_logic_vector": mistake_logic_vector,
                "context_vector": context_vector,
                "extra": {},
            }
            events.append(event)
    
    return events
