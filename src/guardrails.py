from src.config import NEGATIVE

PERMITIDOS = {"recomendar", "buscar_cancion", "buscar_album"}


def fuera_de_dominio(intent: str, hits: list) -> bool:
    return intent not in PERMITIDOS or not hits


def formato_recomendacion(items):
    lines = []
    for it in items:
        m = it["meta"]
        lines.append(f"{m['cancion']} - {m['artista']} ({m['album']}, {m['anio']}) | Tags: {m['tags_animo']}")
    return "\n".join(lines)
