from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class EventoIn(BaseModel):
    evento_id: UUID
    dispositivo_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    clase_objeto: str = Field(min_length=1, max_length=32)
    direccion: str = Field(min_length=1, max_length=16)
    confianza: float = Field(ge=0.0, le=1.0)


class ResultadoIngesta(BaseModel):
    recibidos: int
    insertados: int


class ConteoHora(BaseModel):
    hora: datetime
    clase_objeto: str
    total: int
