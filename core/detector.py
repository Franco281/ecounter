import os
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class CajaPrediccion(BaseModel):
    """Representa una caja candidata devuelta por el detector antes de filtrarla y convertirla en un objeto útil para el conteo."""
    x1: float
    y1: float
    x2: float
    y2: float
    confianza: float = Field(..., ge=0.0, le=1.0)
    clase_id: int


class ObjetoDetectado(BaseModel):
    """Es la estructura final que se entrega al sistema de tracking y conteo: clase, confianza y ubicación exacta del objeto detectado."""
    id_tracking: int | None = None
    clase: str
    confianza: float
    centroide: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]


class DetectorONNX:
    """
    Implementación estándar para Jetson:
    - exportar el modelo YOLO a TensorRT (.engine)
    - cargarlo con ultralytics
    - usar ByteTrack por tracking
    - filtrar clases de interés: peatón, bicicleta, motociclista
    """

    def __init__(self, model_path: str, confianza_min: float = 0.40, device: str = "cuda:0"):
        """Carga el modelo YOLO exportado a TensorRT y prepara la inferencia para que el sistema pueda detectar bicicletas, peatones y motociclistas en cada frame."""
        self.model_path = model_path
        self.confianza_min = confianza_min
        self.device = device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No existe el modelo TensorRT en: {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Falta la dependencia 'ultralytics'. Instálala en el Jetson: pip install ultralytics"
            ) from exc

        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.model.overrides["conf"] = self.confianza_min
        self.model.overrides["device"] = self.device

        self.CLASES_MOVILIDAD: Dict[int, str] = {
            0: "peaton",
            1: "bicicleta",
            3: "motociclista",
        }

    def detectar(self, frame) -> List[ObjetoDetectado]:
        """Ejecuta la inferencia sobre el frame actual y devuelve solo los objetos de interés con su centroide y su ID de seguimiento."""
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confianza_min,
            classes=list(self.CLASES_MOVILIDAD.keys()),
            device=self.device,
            verbose=False,
            stream=False,
        )

        if not results or len(results) == 0:
            return []

        detecciones: List[ObjetoDetectado] = []
        pred = results[0]
        boxes = getattr(pred, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id not in self.CLASES_MOVILIDAD:
                continue

            confianza = float(box.conf.item())
            if confianza < self.confianza_min:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            centro_x = int((x1 + x2) / 2)
            centro_y = int((y1 + y2) / 2)
            track_id = None
            if getattr(box, "id", None) is not None:
                try:
                    track_id = int(box.id.item())
                except (ValueError, AttributeError):
                    track_id = None

            detecciones.append(
                ObjetoDetectado(
                    id_tracking=track_id,
                    clase=self.CLASES_MOVILIDAD[cls_id],
                    confianza=confianza,
                    centroide=(centro_x, centro_y),
                    bounding_box=(x1, y1, x2, y2),
                )
            )

        return detecciones


DetectorYOLOTRT = DetectorONNX
