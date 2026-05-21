import pickle
import threading
from pathlib import Path
import numpy as np


class EmbeddingDB:
    """Tiny in-memory embedding store. Brute-force cosine match.

    For 1-20 enrolled people this is faster than any index.
    Swap to FAISS only if the DB grows past ~500.

    Thread-safe: the HTTP thread may add/remove identities while the
    inference thread is matching. RLock guards both sides.
    """

    def __init__(self, path: Path):
        self.path = path
        self.names: list[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        self._lock = threading.RLock()
        if path.exists():
            self.load()

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        with self._lock:
            self.names = data["names"]
            self.embeddings = data["embeddings"]

    def save(self):
        with self._lock:
            payload = {"names": list(self.names), "embeddings": self.embeddings.copy()}
        with open(self.path, "wb") as f:
            pickle.dump(payload, f)

    def add(self, name: str, embedding: np.ndarray):
        emb = embedding.astype(np.float32).reshape(1, -1)
        emb /= np.linalg.norm(emb) + 1e-9
        with self._lock:
            self.names.append(name)
            self.embeddings = np.vstack([self.embeddings, emb])

    def remove(self, name: str) -> int:
        """Remove all embeddings for a person. Returns count removed."""
        with self._lock:
            keep = [i for i, n in enumerate(self.names) if n != name]
            removed = len(self.names) - len(keep)
            if removed:
                self.names = [self.names[i] for i in keep]
                self.embeddings = self.embeddings[keep] if keep else np.zeros((0, 512), dtype=np.float32)
        return removed

    def match(self, embedding: np.ndarray, threshold: float):
        with self._lock:
            if len(self.names) == 0:
                return None, 0.0
            embs = self.embeddings
            names = self.names
        q = embedding.astype(np.float32).reshape(-1)
        q /= np.linalg.norm(q) + 1e-9
        sims = embs @ q
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= threshold:
            return names[idx], score
        return None, score

    def counts_by_name(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for n in self.names:
                out[n] = out.get(n, 0) + 1
        return out
