import json
import sqlite3
import threading
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .models import EventoCiclovia


class EnviadorEventos:
    """
    Envía eventos al backend sin bloquear el bucle de video (store-and-forward).
    - __call__ guarda el evento en una cola SQLite local y retorna de inmediato.
    - Un hilo en segundo plano envía lotes con POST {url}/api/v1/eventos (cuerpo: lista JSON).
    - Si falla la red o el backend (5xx), reintenta con backoff exponencial; los eventos no se pierden.
    - El backend debe tratar `evento_id` como clave idempotente.
    """

    RUTA = "/api/v1/eventos"

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        db_path: str = "cola_eventos.db",
        lote: int = 50,
        timeout: float = 10.0,
        backoff_max: float = 60.0,
    ):
        """Abre la cola persistente y arranca el hilo de envío."""
        if urlparse(base_url).scheme not in ("http", "https"):
            raise ValueError("base_url debe empezar por http:// o https://")

        self.url = base_url.rstrip("/") + self.RUTA
        self.token = token
        self.lote = lote
        self.timeout = timeout
        self.backoff_max = backoff_max

        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cola ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "evento_id TEXT UNIQUE NOT NULL, "
            "payload TEXT NOT NULL)"
        )
        self._db.commit()

        self._hay_trabajo = threading.Event()
        self._detener = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="enviador-eventos", daemon=True)
        self._hilo.start()
        self._hay_trabajo.set()  # envía lo que haya quedado de ejecuciones anteriores

    def __call__(self, evento: EventoCiclovia) -> None:
        """Se usa como api_callback: encola el evento y despierta al hilo."""
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO cola (evento_id, payload) VALUES (?, ?)",
                (evento.evento_id, evento.model_dump_json()),
            )
            self._db.commit()
        self._hay_trabajo.set()

    def cerrar(self, espera: float = 5.0) -> None:
        """Intenta vaciar la cola un tiempo limitado y detiene el hilo; lo pendiente queda guardado en disco."""
        self._hay_trabajo.set()
        self._hilo.join(timeout=espera)
        self._detener.set()
        self._hay_trabajo.set()
        self._hilo.join(timeout=2.0)
        with self._lock:
            self._db.close()

    def _pendientes(self) -> list[tuple[int, str]]:
        with self._lock:
            return self._db.execute(
                "SELECT id, payload FROM cola ORDER BY id LIMIT ?", (self.lote,)
            ).fetchall()

    def _borrar(self, ids: list[int]) -> None:
        with self._lock:
            self._db.executemany("DELETE FROM cola WHERE id = ?", [(i,) for i in ids])
            self._db.commit()

    def _bucle(self) -> None:
        espera = 1.0
        while not self._detener.is_set():
            self._hay_trabajo.wait()
            self._hay_trabajo.clear()

            while not self._detener.is_set():
                filas = self._pendientes()
                if not filas:
                    espera = 1.0
                    break

                resultado = self._enviar([json.loads(p) for _, p in filas])
                if resultado == "ok" or resultado == "descartar":
                    self._borrar([i for i, _ in filas])
                    espera = 1.0
                else:
                    if self._detener.wait(espera):
                        return
                    espera = min(espera * 2, self.backoff_max)

    def _enviar(self, eventos: list[dict]) -> str:
        """Devuelve 'ok', 'reintentar' (red/5xx/429) o 'descartar' (rechazo definitivo 4xx)."""
        peticion = urllib.request.Request(
            self.url,
            data=json.dumps(eventos).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.token:
            peticion.add_header("Authorization", f"Bearer {self.token}")

        try:
            with urllib.request.urlopen(peticion, timeout=self.timeout) as resp:
                return "ok" if 200 <= resp.status < 300 else "reintentar"
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429) or exc.code >= 500:
                return "reintentar"
            print(f"Backend rechazo el lote (HTTP {exc.code}); se descarta.")
            return "descartar"
        except (urllib.error.URLError, TimeoutError, OSError):
            return "reintentar"
