from typing import Tuple

from pydantic import BaseModel


class DeteccionObjeto(BaseModel):
    id_tracking: int
    clase: str
    confianza: float
    centroide: Tuple[int, int]


class EventoCiclovia(BaseModel):
    timestamp: str
    clase_objeto: str
    direccion: str
    confianza: float
