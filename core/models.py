from typing import Tuple

import uuid

from pydantic import BaseModel, Field


class DeteccionObjeto(BaseModel):
    id_tracking: int
    clase: str
    confianza: float
    centroide: Tuple[int, int]


class EventoCiclovia(BaseModel):
    evento_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dispositivo_id: str = ""
    timestamp: str
    clase_objeto: str
    direccion: str
    confianza: float
