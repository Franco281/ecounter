import json
from typing import List, Tuple

import cv2
from pydantic import BaseModel


class ConfiguracionEscena(BaseModel):
    """Define la zona útil de observación y la línea de conteo para cada instalación."""
    roi: Tuple[int, int, int, int] = (0, 0, 1280, 720)
    linea_conteo: Tuple[int, int, int, int] = (0, 300, 1280, 300)
    direccion_aceptada: str = "sur"

    @classmethod
    def desde_json(cls, path: str):
        """Carga la geometría del sitio desde un archivo JSON para reutilizar la misma calibración en otra instalación."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(**payload)

    def guardar_json(self, path: str):
        """Guarda la zona de observación y la línea de conteo para reutilizarla sin volver a calibrar la cámara."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.model_dump(), fh, indent=2)


def calibrar_escena(video_source: str | int = 0, output_path: str = "configuracion_escena.json") -> ConfiguracionEscena:
    """
    Abre la cámara y permite ajustar a mano el rectángulo de observación y la línea de conteo.
    Esta calibración es clave porque cada ubicación tiene una perspectiva distinta y una línea fija en píxeles no sirve en todas las instalaciones.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {video_source}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("No se pudo leer ningún fotograma para calibrar la escena.")

    roi_points: List[Tuple[int, int]] = []
    line_points: List[Tuple[int, int]] = []
    etapa = "roi"
    cursor = (0, 0)

    def mouse_callback(event, x, y, flags, param):
        """Captura los clics del usuario para definir los puntos del ROI y de la línea de conteo sobre la imagen de la cámara."""
        nonlocal roi_points, line_points, etapa, cursor
        cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if etapa == "roi" and len(roi_points) < 2:
                roi_points.append((x, y))
            elif etapa == "linea" and len(line_points) < 2:
                line_points.append((x, y))

    cv2.namedWindow("Calibracion de escena", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibracion de escena", mouse_callback)

    while True:
        imagen = frame.copy()

        if roi_points:
            esquina = roi_points[1] if len(roi_points) == 2 else cursor
            cv2.rectangle(imagen, roi_points[0], esquina, (0, 255, 0), 2)
            cv2.circle(imagen, roi_points[0], 5, (0, 255, 0), -1)

        if line_points:
            fin = line_points[1] if len(line_points) == 2 else cursor
            cv2.line(imagen, line_points[0], fin, (0, 0, 255), 2)
            cv2.circle(imagen, line_points[0], 5, (0, 0, 255), -1)

        instrucciones = [
            "Calibracion del ROI y la linea de conteo",
            "1) click en dos esquinas para el ROI",
            "2) pulse 'l' y haga click en dos puntos para la linea",
            "3) pulse 's' para guardar JSON",
            "4) pulse 'q' para salir",
            "5) pulse 'r' para reiniciar",
            f"Etapa actual: {'ROI' if etapa == 'roi' else 'LINEA'}",
        ]
        alto_panel = 12 + len(instrucciones) * 24
        cv2.rectangle(imagen, (0, 0), (480, alto_panel), (0, 0, 0), -1)
        for i, txt in enumerate(instrucciones):
            cv2.putText(imagen, txt, (10, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Calibracion de escena", imagen)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("l") and len(roi_points) == 2:
            etapa = "linea"
        elif key == ord("r"):
            roi_points = []
            line_points = []
            etapa = "roi"
        elif key == ord("s"):
            if len(roi_points) == 2 and len(line_points) == 2:
                x1, y1 = roi_points[0]
                x2, y2 = roi_points[1]
                x3, y3 = line_points[0]
                x4, y4 = line_points[1]

                config = ConfiguracionEscena(
                    roi=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                    linea_conteo=(x3, y3, x4, y4),
                    direccion_aceptada="sur",
                )
                config.guardar_json(output_path)
                print(f"Configuracion guardada en: {output_path}")
                cap.release()
                cv2.destroyAllWindows()
                return config
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Calibracion cancelada por el usuario.")
