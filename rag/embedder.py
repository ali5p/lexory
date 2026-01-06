from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

