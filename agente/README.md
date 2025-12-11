# Agente musical (Flask + Gemini)

## Requisitos
- Python 3.10+
- Clave de Gemini: define `GEMINI_API_KEY` (y opcional `GEMINI_MODEL`, ej. `gemini-1.5-flash`).

## Instalación
```bash
cd agente
pip install -r requirements.txt
```

## Configuración
- Copia `.env.example` a `.env` o exporta la variable:
  - `GEMINI_API_KEY=tu_clave`
  - Opcional: `TOP_K`, `MAX_ITEMS`, `NEGATIVE`, `DATA_*` si cambias rutas.

## Ejecutar
```bash
cd agente
python app.py
# abre http://localhost:5000
```

## Estructura
- `app.py`: servidor Flask y endpoint `/api/chat`.
- `config.py`: carga variables de entorno.
- `model/agente.py`: RAG ligero + guardrails + llamada a Gemini.
- `data/`: catálogo, políticas, FAQ (editable).
- `templates/`, `static/`: interfaz web.

## Notas
- Si no defines `GEMINI_API_KEY`, el agente responde con el fallback basado en TF-IDF y el catálogo local.
- Ajusta el catálogo (`agente/data/catalogo.csv`) y políticas/FAQ según tu institución.***
