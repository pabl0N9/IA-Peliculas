import pandas as pd
from pathlib import Path
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Agente:
    def __init__(self, settings):
        self.settings = settings
        self.politicas = Path(settings.data_politicas).read_text(encoding="utf-8")
        self.faq = Path(settings.data_faq).read_text(encoding="utf-8")
        self.df = pd.read_csv(settings.data_catalog)
        self.artistas = self._load_artistas(Path(settings.data_artistas))
        self.docs = self._build_docs(self.df)
        self.retriever = self._build_retriever(self.docs)
        self._init_model()

    def _init_model(self):
        if not self.settings.gemini_api_key:
            self.model = None
            return
        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(self.settings.model_name)

    def _load_artistas(self, path: Path):
        if not path.exists():
            return {}
        df = pd.read_csv(path, encoding="latin-1")
        mapping = {}
        for _, r in df.iterrows():
            mapping[str(r.artist).lower()] = {
                "bio": str(r.bio),
                "songs": [s.strip() for s in str(r.top_songs).split(";") if s.strip()],
            }
        return mapping

    def _build_docs(self, df: pd.DataFrame):
        docs = []
        for _, r in df.iterrows():
            text = f"{r.cancion} - {r.artista} ({r.album}, {r.anio}). Genero: {r.genero}. Tags: {r.tags_animo} {r.tags_tema}"
            docs.append({"id": int(r.id), "text": text, "meta": r.to_dict()})
        return docs

    def _build_retriever(self, docs):
        texts = [d["text"] for d in docs]
        metas = [{"text": d["text"], "meta": d["meta"]} for d in docs]
        vect = TfidfVectorizer()
        matrix = vect.fit_transform(texts)
        return {"vectorizer": vect, "matrix": matrix, "metas": metas}

    def _search(self, query, k=None):
        k = k or self.settings.top_k
        vect = self.retriever["vectorizer"]
        matrix = self.retriever["matrix"]
        metas = self.retriever["metas"]
        qv = vect.transform([query])
        sims = cosine_similarity(qv, matrix).flatten()
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            if sims[i] <= 0:
                continue
            results.append({"text": metas[i]["text"], "meta": metas[i]["meta"], "score": float(sims[i])})
        return results

    def _artist_match(self, pregunta: str):
        q = pregunta.lower()
        for nombre, info in self.artistas.items():
            if nombre in q:
                return nombre, info
        return None, None

    def _formato_recomendacion(self, items):
        lines = []
        for it in items[: self.settings.max_items]:
            m = it["meta"]
            tags = str(m.get("tags_animo", "")).replace(";", ", ")
            lines.append(f"{m['cancion']} - {m['artista']} ({m['album']}, {m['anio']}) | {tags}")
        return "\n".join(lines)

    def _llm_summarize(self, pregunta: str, contexto: str) -> str | None:
        if not self.model:
            return None
        prompt = f"""Eres un asistente musical institucional. Responde en 2-3 frases claras y faciles de leer, sin listas ni etiquetas, usando solo este contexto.
Contexto:
{contexto}

Pregunta del usuario: {pregunta}

Si el contexto no contiene la respuesta, responde exactamente: {self.settings.negative}"""
        try:
            result = self.model.generate_content(prompt)
            text = getattr(result, "text", "").strip()
            return text if text else None
        except Exception:
            return None

    def build_prompt(self, pregunta: str, hits: list) -> str:
        contexto = "\n".join([h["text"] for h in hits])
        prompt = f"""Eres un asistente musical institucional. Usa solo el contexto.
Politicas:
{self.politicas}

FAQ:
{self.faq}

Contexto:
{contexto}

Usuario: {pregunta}
Responde en tono breve. Si no hay datos suficientes, responde exactamente: {self.settings.negative}"""
        return prompt

    def generate_response(self, pregunta: str) -> str:
        # 1) Artista: bio + canciones clave, resumido con LLM si existe
        nombre, info = self._artist_match(pregunta)
        if nombre:
            bio = info["bio"]
            songs = info.get("songs", [])
            if not songs:
                canciones = self.df[self.df["artista"].str.lower() == nombre]["cancion"].tolist()
                songs = canciones[: self.settings.max_items]
            songs_lines = "\n".join([f"{i+1}. {s}" for i, s in enumerate(songs)]) if songs else "Sin canciones listadas"
            if "cancion" in pregunta.lower() or "canciones" in pregunta.lower():
                return f"Canciones de {nombre.title()}:\n{songs_lines}"
            contexto = f"Artista: {nombre.title()}. Bio: {bio}."
            summary = self._llm_summarize(pregunta, contexto)
            cuerpo = summary if summary else f"Artista: {nombre.title()}. Bio: {bio}."
            return f"{cuerpo}\n\nCanciones:\n{songs_lines}"

        # 2) Recupera canciones y responde
        hits = self._search(pregunta, k=self.settings.top_k)
        if not hits:
            return self.settings.negative

        # 3) Si hay modelo, usa Gemini con contexto; si falla, usa fallback formateado
        contexto = "\n".join([h["text"] for h in hits])
        summary = self._llm_summarize(pregunta, contexto)
        if summary:
            return summary
        return self._formato_recomendacion(hits)
