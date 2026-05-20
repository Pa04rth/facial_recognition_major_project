import pickle
from pathlib import Path
import numpy as np


class EmbeddingDB:
    """Tiny in-memory embedding store. Brute-force cosine match.

    For 1-20 enrolled people this is faster than any index.
    Swap to FAISS only if the DB grows past ~500.
    """

    def __init__(self, path: Path):
        self.path = path
        self.names: list[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        if path.exists():
            self.load()

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.names = data["names"]
        self.embeddings = data["embeddings"]

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump({"names": self.names, "embeddings": self.embeddings}, f)

    def add(self, name: str, embedding: np.ndarray):
        emb = embedding.astype(np.float32).reshape(1, -1)
        emb /= np.linalg.norm(emb) + 1e-9
        self.names.append(name)
        self.embeddings = np.vstack([self.embeddings, emb])

    def match(self, embedding: np.ndarray, threshold: float):
        if len(self.names) == 0:
            return None, 0.0
        q = embedding.astype(np.float32).reshape(-1)
        q /= np.linalg.norm(q) + 1e-9
        sims = self.embeddings @ q
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= threshold:
            return self.names[idx], score
        return None, score
