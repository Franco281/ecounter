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
        try:
            import torch
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"
        self.device = device

        if model_path.endswith((".engine", ".onnx")) and not os.path.exists(self.model_path):
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

    def detectar(self, frame, roi: Tuple[int, int, int, int] | None = None, imgsz: int = 960) -> List[ObjetoDetectado]:
        """Ejecuta la inferencia sobre el ROI del frame (recortado, para ganar resolución efectiva) y devuelve solo los objetos de interés en coordenadas del frame completo."""
        off_x, off_y = 0, 0
        entrada = frame
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            h, w = frame.shape[:2]
            rx1, ry1, rx2, ry2 = max(0, rx1), max(0, ry1), min(w, rx2), min(h, ry2)
            if rx2 > rx1 and ry2 > ry1:
                entrada = frame[ry1:ry2, rx1:rx2]
                off_x, off_y = rx1, ry1

        results = self.model.track(
            entrada,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confianza_min,
            imgsz=imgsz,
            iou=0.5,
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
            x1, x2 = x1 + off_x, x2 + off_x
            y1, y2 = y1 + off_y, y2 + off_y
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

        return self._suprimir_duplicados(detecciones)

    @staticmethod
    def _suprimir_duplicados(detecciones: List[ObjetoDetectado], umbral: float = 0.6) -> List[ObjetoDetectado]:
        """Descarta cajas de la misma clase que se solapan mucho (IoU o contención), conservando la de mayor confianza."""
        conservadas: List[ObjetoDetectado] = []
        for d in sorted(detecciones, key=lambda x: x.confianza, reverse=True):
            ax1, ay1, ax2, ay2 = d.bounding_box
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            duplicado = False
            for k in conservadas:
                if k.clase != d.clase:
                    continue
                bx1, by1, bx2, by2 = k.bounding_box
                iw = min(ax2, bx2) - max(ax1, bx1)
                ih = min(ay2, by2) - max(ay1, by1)
                if iw <= 0 or ih <= 0:
                    continue
                inter = iw * ih
                area_b = max(1, (bx2 - bx1) * (by2 - by1))
                if inter / min(area_a, area_b) > umbral or inter / (area_a + area_b - inter) > 0.4:
                    duplicado = True
                    break
            if not duplicado:
                conservadas.append(d)
        return conservadas


DetectorYOLOTRT = DetectorONNX
