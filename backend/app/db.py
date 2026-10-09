from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False) if engine else None


def get_db():
    """Dependencia de FastAPI: una sesión por petición."""
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL no configurada; ejecute python -m app.init_db")
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
