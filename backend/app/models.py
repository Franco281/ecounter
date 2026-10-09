import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EventoConteo(Base):
    __tablename__ = "eventos_conteo"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    evento_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True, nullable=False)
    dispositivo_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    clase_objeto: Mapped[str] = mapped_column(String(32), nullable=False)
    direccion: Mapped[str] = mapped_column(String(16), nullable=False)
    confianza: Mapped[float] = mapped_column(Float, nullable=False)
    recibido_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_eventos_dispositivo_timestamp", "dispositivo_id", "timestamp"),
        Index("ix_eventos_timestamp", "timestamp"),
    )
