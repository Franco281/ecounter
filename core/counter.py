import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from pydantic import BaseModel

from .config import ConfiguracionEscena
from .detector import DetectorONNX
from .events import imprimir_evento
from .models import DeteccionObjeto, EventoCiclovia


class ContadorMovilidadPerimetral:
    """
    Capa de aplicación para un sistema edge con cámara.
    - usa YOLO exportado a TensorRT si existe un engine
    - usa ByteTrack como tracker en el detector
    - ejecuta conteo por línea virtual
    - si no hay modelo, cae a simulación para pruebas locales
    """

    def __init__(
        self,
        video_source: str | int,
        api_callback: Callable[[EventoCiclovia], None],
        model_path: str | None = None,
        linea_conteo_y: int | None = None,
        confianza_min: float = 0.40,
        config: ConfiguracionEscena | dict | None = None,
        config_path: str | None = None,
    ):
        """Inicializa la cámara, la configuración del sitio y el detector; si no hay engine real, el sistema entra en modo de demostración para pruebas locales."""
        if video_source == "":
            video_source = 0

        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {video_source}")

        if config_path is not None:
            config = ConfiguracionEscena.desde_json(config_path)
        elif config is not None and isinstance(config, dict):
            config = ConfiguracionEscena(**config)
        elif config is None:
            config = ConfiguracionEscena()

        self.config = config
        self.api_callback = api_callback
        self.roi = self.config.roi
        self.linea_conteo = self.config.linea_conteo
        self.linea_conteo_y = self.linea_conteo[1]
        if linea_conteo_y is not None:
            self.linea_conteo_y = linea_conteo_y
            self.linea_conteo = (0, linea_conteo_y, 1280, linea_conteo_y)
        self.confianza_min = confianza_min
        self.detector: DetectorONNX | None = None

        if model_path:
            try:
                self.detector = DetectorONNX(model_path=model_path, confianza_min=self.confianza_min)
            except Exception as exc:
                print(f"Advertencia: no se pudo cargar el modelo TensorRT: {exc}")
                self.detector = None

        self.historial_posiciones: Dict[int, Tuple[int, int]] = {}
        self.ids_contados: set[int] = set()
        self._tracking_actual: Dict[int, Tuple[int, int]] = {}
        self.next_tracking_id = 1

    def detectar(self, frame: np.ndarray) -> List[DeteccionObjeto]:
        """Obtiene las detecciones del frame actual y las normaliza para que la lógica de conteo trabaje con una estructura uniforme, independientemente de si usa detector real o simulación."""
        if self.detector is not None:
            objetos = self.detector.detectar(frame)
            return self._normalizar_detecciones(objetos)
        return self._simular_rastreador_ia(frame)

    def _normalizar_detecciones(self, objetos: List) -> List[DeteccionObjeto]:
        """Convierte la salida del detector a un contrato estable para la lógica de conteo, asegurando que cada objeto tenga una clase, un centroide y un ID útil."""
        resultados: List[DeteccionObjeto] = []
        for obj in objetos:
            id_tracking = obj.id_tracking if obj.id_tracking is not None else self.next_tracking_id
            if obj.id_tracking is None:
                self.next_tracking_id += 1
            resultados.append(
                DeteccionObjeto(
                    id_tracking=int(id_tracking),
                    clase=obj.clase,
                    confianza=float(obj.confianza),
                    centroide=obj.centroide,
                )
            )
        return resultados

    def _simular_rastreador_ia(self, frame: np.ndarray) -> List[DeteccionObjeto]:
        """Genera un objeto virtual en movimiento para probar la lógica de conteo cuando no hay cámara ni modelo real disponible."""
        simulado_y = int((time.time() % 10) * 60)
        if simulado_y > 550:
            return []

        return [
            DeteccionObjeto(
                id_tracking=42,
                clase="bicicleta",
                confianza=0.92,
                centroide=(320, simulado_y),
            )
        ]

    def _cruza_linea(self, centroide: Tuple[int, int], y_anterior: int | None = None) -> bool:
        """Decide si el centroide del objeto ha pasado la línea de conteo; esto es lo que dispara el evento de entrada o salida del flujo."""
        x_actual, y_actual = centroide
        x1, y1, x2, y2 = self.linea_conteo

        if abs(y2 - y1) < 2:
            if y_anterior is None:
                return False
            return y_anterior <= self.linea_conteo_y < y_actual

        if abs(x2 - x1) < 2:
            if y_anterior is None:
                return False
            return x_actual >= min(x1, x2) and x_actual <= max(x1, x2) and y_anterior <= y_actual

        return False

    def procesar_video(self):
        """Recorre la secuencia de video, evalúa la posición de cada objeto y dispara un evento cada vez que cruza la línea de conteo."""
        fps_limiter = 0
        frame_skip = 2

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break

            fps_limiter += 1
            if fps_limiter % frame_skip != 0:
                continue

            objetos_detectados = self.detectar(frame)

            for obj in objetos_detectados:
                id_obj = obj.id_tracking
                x_actual, y_actual = obj.centroide

                if id_obj not in self.historial_posiciones:
                    self.historial_posiciones[id_obj] = (x_actual, y_actual)
                    continue

                x_anterior, y_anterior = self.historial_posiciones[id_obj]
                self.historial_posiciones[id_obj] = (x_actual, y_actual)

                if self._cruza_linea((x_actual, y_actual), y_anterior):
                    if id_obj not in self.ids_contados:
                        self.ids_contados.add(id_obj)
                        evento = EventoCiclovia(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            clase_objeto=obj.clase,
                            direccion=self.config.direccion_aceptada,
                            confianza=obj.confianza,
                        )
                        self.api_callback(evento)

            if len(self.historial_posiciones) > 200:
                self._limpiar_memoria_tracking()

        self.video.release()

    def _limpiar_memoria_tracking(self):
        """Evita que el historial de objetos crezca indefinidamente y consuma memoria en una ejecución continua del sistema."""
        self.historial_posiciones.clear()
        self.ids_contados.clear()
        self._tracking_actual.clear()


# compatibility alias
Counter = ContadorMovilidadPerimetral
