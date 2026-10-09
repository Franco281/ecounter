"""Crea el rol, la base de datos y las tablas, y genera backend/.env.

Uso:  python -m app.init_db
La contrasena del superusuario se pide por teclado (o variable PGADMIN_PASSWORD) y no se guarda.
"""
import os
import secrets
from getpass import getpass

import psycopg
from psycopg import sql
from sqlalchemy import create_engine

from .config import DB_NAME, DB_USER, ENV_PATH
from .models import Base


def main() -> None:
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    admin_user = os.environ.get("PGADMIN_USER", "postgres")
    admin_password = os.environ.get("PGADMIN_PASSWORD") or getpass(f"Contrasena de '{admin_user}' en PostgreSQL: ")

    app_password = secrets.token_urlsafe(24)

    with psycopg.connect(host=host, port=port, user=admin_user, password=admin_password, dbname="postgres", autocommit=True) as conn:
        existe_rol = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (DB_USER,)).fetchone()
        verbo = "ALTER ROLE {} WITH LOGIN PASSWORD {}" if existe_rol else "CREATE ROLE {} LOGIN PASSWORD {}"
        conn.execute(sql.SQL(verbo).format(sql.Identifier(DB_USER), sql.Literal(app_password)))

        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8'").format(sql.Identifier(DB_NAME), sql.Identifier(DB_USER)))
            print(f"Base de datos creada: {DB_NAME}")
        else:
            print(f"La base de datos {DB_NAME} ya existe.")

    url = f"postgresql+psycopg://{DB_USER}:{app_password}@{host}:{port}/{DB_NAME}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    print("Tablas creadas/verificadas.")

    token = secrets.token_urlsafe(32)
    ENV_PATH.write_text(f"DATABASE_URL={url}\nAPI_TOKEN={token}\n", encoding="utf-8")
    print(f"Configuracion escrita en {ENV_PATH}")
    print("Para el edge:  $env:BACKEND_TOKEN = (token de API_TOKEN en backend\\.env)")


if __name__ == "__main__":
    main()
