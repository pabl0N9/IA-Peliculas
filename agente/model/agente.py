import pandas as pd
from pathlib import Path
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Agente:
    def __init__(self, settings):
        self.settings = settings
        self.politicas = Path(settings.data_politicas).read_text(encoding="latin-1")
        self.faq = Path(settings.data_faq).read_text(encoding="latin-1")
        self.df = pd.read_csv(settings.data_catalog, encoding="latin-1")
        self.actores = self._load_actores(Path(settings.data_artistas))
        self.docs = self._build_docs(self.df)
        self.retriever = self._build_retriever(self.docs)
        self._init_model()

    def _init_model(self):
        if not self.settings.gemini_api_key:
            self.model = None
            return
        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(self.settings.model_name)

    def _load_actores(self, path: Path):
        if not path.exists():
            return {}
        df = pd.read_csv(path, encoding="latin-1")
        mapping = {}
        for _, r in df.iterrows():
            mapping[str(r.actor).lower()] = {
                "bio": str(r.bio),
                "movies": [s.strip() for s in str(r.top_movies).split(";") if s.strip()],
            }
        return mapping

    def _build_docs(self, df: pd.DataFrame):
        docs = []
        for _, r in df.iterrows():
            text = f"{r.titulo} ({r.anio}). Genero: {r.genero}. Elenco: {r.elenco}. Sinopsis: {r.descripcion}. Tags: {r.tags}"
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

    def _actor_match(self, pregunta: str):
        q = pregunta.lower()
        for nombre, info in self.actores.items():
            if nombre in q:
                return nombre, info
        return None, None

    def _formato_recomendacion(self, items, max_count=None):
        limite = max_count if max_count is not None else self.settings.max_items
        lines = []
        for it in items[:limite]:
            m = it["meta"]
            tags = str(m.get("tags", "")).replace(";", ", ")
            elenco = m.get("elenco", "")
            lines.append(f"{m['titulo']} ({m['anio']}) - {m['genero']} | Elenco: {elenco} | {tags}")
        return "\n".join(lines)

    def _llm_summarize(self, pregunta: str, contexto: str) -> str | None:
        if not self.model:
            return None
        prompt = f"""Eres un asistente de peliculas institucional. Responde en 2-3 frases claras y faciles de leer, sin listas ni etiquetas, usando solo este contexto.
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
        prompt = f"""Eres un asistente de peliculas institucional. Usa solo el contexto.
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
        nombre, info = self._actor_match(pregunta)
        if nombre:
            bio = info["bio"]
            movies = info.get("movies", [])
            if not movies:
                peliculas = self.df[self.df["elenco"].str.lower().str.contains(nombre, na=False)]["titulo"].tolist()
                movies = peliculas[:3]
            movies = movies[:3]
            movies_lines = "\n".join([f"{i+1}. {s}" for i, s in enumerate(movies)]) if movies else "Sin peliculas listadas"
            if "pelicula" in pregunta.lower() or "peliculas" in pregunta.lower():
                return f"Peliculas de {nombre.title()}:\n{movies_lines}"
            contexto = f"Actor/Actriz: {nombre.title()}. Bio: {bio}."
            summary = self._llm_summarize(pregunta, contexto)
            cuerpo = summary if summary else f"Actor/Actriz: {nombre.title()}. Bio: {bio}."
            return f"{cuerpo}\n\nPeliculas:\n{movies_lines}"

        keyword_pelis = "pelicula" in pregunta.lower() or "peliculas" in pregunta.lower()
        max_count = 5 if keyword_pelis else self.settings.max_items
        hits = self._search(pregunta, k=max(max_count, self.settings.top_k))
        if not hits:
            if keyword_pelis:
                fallback_items = []
                for _, r in self.df.head(max_count).iterrows():
                    fallback_items.append({"meta": r.to_dict(), "text": "", "score": 0.0})
                return self._formato_recomendacion(fallback_items, max_count=max_count)
            return self.settings.negative

        contexto = "\n".join([h["text"] for h in hits])
        summary = self._llm_summarize(pregunta, contexto)
        if summary:
            return summary
        return self._formato_recomendacion(hits, max_count=max_count)
