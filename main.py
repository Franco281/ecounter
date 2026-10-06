from core.config import ConfiguracionEscena, calibrar_escena
from core.counter import ContadorMovilidadPerimetral
from core.events import imprimir_evento
from tkinter import Tk, filedialog


def seleccionar_video() -> str:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ruta = filedialog.askopenfilename(
        title="Seleccionar video",
        filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Todos", "*.*")],
    )
    root.destroy()
    return ruta


if __name__ == "__main__":
    """Punto de entrada principal del sistema: carga la escena configurada o la calibra automáticamente y arranca el flujo de video, detección y conteo."""
    video_source = ""
    if video_source == "":
        video_source = seleccionar_video()
        if not video_source:
            print("No se selecciono ningun video.")
            raise SystemExit(0)

    config_path = "configuracion_escena.json"
    try:
        config = ConfiguracionEscena.desde_json(config_path)
        print(f"Usando configuracion cargada desde: {config_path}")
    except FileNotFoundError:
        print(f"No existe {config_path}. Iniciando calibracion interactiva...")
        try:
            config = calibrar_escena(video_source=video_source, output_path=config_path)
        except KeyboardInterrupt:
            print("Calibracion cancelada.")
            raise SystemExit(0)

    contador = ContadorMovilidadPerimetral(
        video_source=video_source,
        api_callback=imprimir_evento,
        model_path=None,
        config=config,
        confianza_min=0.45,
    )
    contador.procesar_video()
