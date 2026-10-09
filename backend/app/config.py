import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
ENV_PATH = RAIZ / ".env"

load_dotenv(ENV_PATH)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
API_TOKEN = os.environ.get("API_TOKEN", "")

DB_NAME = "trafico_ciclista"
DB_USER = "trafico_app"
