from src.config import NEGATIVE
from src.guardrails import fuera_de_dominio, formato_recomendacion


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
        return NEGATIVE
    if llm_callable:
        prompt = render_prompt(politicas, faq, hits, pregunta)
        raw = llm_callable(prompt)
        return raw.strip()
    # Fallback sin LLM: devuelve formato con top-3
    return formato_recomendacion(hits[:3])
