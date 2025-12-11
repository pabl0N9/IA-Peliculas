import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_catalog(path: str):
    """Load CSV catalog as DataFrame."""
    return pd.read_csv(path)


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_docs(df: pd.DataFrame):
    docs = []
    for _, r in df.iterrows():
        text = f"{r.cancion} - {r.artista} ({r.album}, {r.anio}). Genero: {r.genero}. Tags: {r.tags_animo} {r.tags_tema}"
        docs.append({"id": int(r.id), "text": text, "meta": r.to_dict()})
    return docs


class TfidfRetriever:
    """Lightweight retriever; replace with embeddings model if available."""

    def __init__(self, texts, metas):
        self.vectorizer = TfidfVectorizer()
        self.metas = metas
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=5):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            if sims[i] <= 0:
                continue
            results.append({"text": self.metas[i]["text"], "meta": self.metas[i]["meta"], "score": float(sims[i])})
        return results
