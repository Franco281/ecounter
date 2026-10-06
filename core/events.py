from pydantic import BaseModel

from .models import EventoCiclovia


def imprimir_evento(evento: EventoCiclovia) -> None:
    """Saca por consola el evento del conteo para inspección local o para depurar la lógica sin una API conectada."""
    print(evento.model_dump_json())
