"""Deterministic mistake_logic vector generator."""
import hashlib
from typing import List

from rag.utils.assets import load_mistake_logic_vocab


def generate_mistake_logic_vector(mistake_type: str) -> List[float]:
    """
    Generate a deterministic 64-dim vector from mistake_type.
    
    Uses one-hot encoding if vocab size <= 64, otherwise hashes to sparse vector.
    """
    vocab_map = load_mistake_logic_vocab()
    vocab_size = len(vocab_map)
    
    if mistake_type not in vocab_map:
        mistake_type = "other"
    
    category_idx = vocab_map[mistake_type]
    
    if vocab_size <= 64:
        vector = [0.0] * 64
        vector[category_idx] = 1.0
        return vector
    
    # Hash-based sparse vector for vocab > 64
    hash_obj = hashlib.md5(mistake_type.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    target_idx = hash_int % 64
    
    vector = [0.0] * 64
    vector[target_idx] = 1.0
    return vector
