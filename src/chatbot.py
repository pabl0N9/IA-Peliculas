from src.config import NEGATIVE
from src.guardrails import fuera_de_dominio, formato_recomendacion


def _temas_disponibles(retriever) -> list:
    """Return unique allowed topics from the catalog metadata."""
    temas = []
    metas = getattr(getattr(retriever, "engine", None), "metas", [])
    for item in metas:
        raw = item["meta"].get("tags_tema", "")
        if not raw:
            continue
        raw = str(raw).replace(",", ";")
        for t in raw.split(";"):
            topic = t.strip()
            if topic and topic not in temas:
                temas.append(topic)
    return temas


def render_prompt(politicas: str, faq: str, pasajes: list, pregunta: str) -> str:
    ctx = "\n".join([p["text"] for p in pasajes])
    return f"""Eres un asistente musical institucional. Usa solo el contexto.
Politicas:
{politicas}

FAQ:
{faq}

Contexto:
{ctx}

Usuario: {pregunta}
"""


def responder(retriever, politicas: str, faq: str, pregunta: str, intent: str, llm_callable=None):
    hits = retriever.search(pregunta, k=5)
    if fuera_de_dominio(intent, hits):
        temas = _temas_disponibles(retriever)
        lista = "\n- ".join(temas) if temas else "No tengo temas disponibles por ahora."
        return f"Por el momento no tengo respuestas sobre eso, pero aqui te dejo temas de los cuales te puedo responder:\n- {lista}"
    if llm_callable:
        prompt = render_prompt(politicas, faq, hits, pregunta)
        raw = llm_callable(prompt)
        return raw.strip()
    # Fallback sin LLM: devuelve formato con top-3
    return formato_recomendacion(hits[:3])
