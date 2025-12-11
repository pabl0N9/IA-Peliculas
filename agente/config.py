import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_items: int = int(os.getenv("MAX_ITEMS", "3"))
    negative: str = os.getenv("NEGATIVE", "No tengo datos autorizados para responder eso.")
    data_catalog: str = os.getenv("DATA_CATALOG", str(BASE_DIR / "data/catalogo.csv"))
    data_politicas: str = os.getenv("DATA_POLITICAS", str(BASE_DIR / "data/politicas.md"))
    data_faq: str = os.getenv("DATA_FAQ", str(BASE_DIR / "data/faq.md"))
    data_artistas: str = os.getenv("DATA_ARTISTAS", str(BASE_DIR / "data/artistas.csv"))
