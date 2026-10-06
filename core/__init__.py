"""Módulos núcleo para el sistema de conteo de movilidad."""

from .config import ConfiguracionEscena, calibrar_escena
from .models import DeteccionObjeto, EventoCiclovia
from .detector import DetectorONNX, DetectorYOLOTRT, CajaPrediccion, ObjetoDetectado
from .counter import ContadorMovilidadPerimetral, imprimir_evento

__all__ = [
    "ConfiguracionEscena",
    "calibrar_escena",
    "DeteccionObjeto",
    "EventoCiclovia",
    "CajaPrediccion",
    "ObjetoDetectado",
    "DetectorONNX",
    "DetectorYOLOTRT",
    "ContadorMovilidadPerimetral",
    "imprimir_evento",
]
