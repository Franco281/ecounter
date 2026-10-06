from core.config import ConfiguracionEscena, calibrar_escena
from core.counter import ContadorMovilidadPerimetral
from core.events import imprimir_evento
from core.models import DeteccionObjeto, EventoCiclovia

__all__ = [
    "ConfiguracionEscena",
    "calibrar_escena",
    "DeteccionObjeto",
    "EventoCiclovia",
    "ContadorMovilidadPerimetral",
    "imprimir_evento",
]


if __name__ == "__main__":
    config_path = "configuracion_escena.json"
    try:
        config = ConfiguracionEscena.desde_json(config_path)
        print(f"Usando configuracion cargada desde: {config_path}")
    except FileNotFoundError:
        print(f"No existe {config_path}. Iniciando calibracion interactiva...")
        try:
            config = calibrar_escena(video_source="", output_path=config_path)
        except KeyboardInterrupt:
            print("Calibracion cancelada.")
            raise SystemExit(0)

    contador = ContadorMovilidadPerimetral(
        video_source="",
        api_callback=imprimir_evento,
        model_path=None,
        config=config,
        confianza_min=0.45,
    )
    contador.procesar_video()