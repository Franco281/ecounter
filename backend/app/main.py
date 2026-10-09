import secrets
from datetime import datetime
from typing import Annotated

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .config import API_TOKEN
from .db import get_db
from .models import EventoConteo
from .schemas import ConteoHora, EventoIn, ResultadoIngesta

app = FastAPI(title="Trafico Ciclista - Backend")


def autenticar(authorization: Annotated[str | None, Header()] = None) -> None:
    """Exige 'Authorization: Bearer <API_TOKEN>' en todos los endpoints protegidos."""
    esperado = f"Bearer {API_TOKEN}"
    if not API_TOKEN or not authorization or not secrets.compare_digest(authorization, esperado):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token invalido")


@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"estado": "ok"}


@app.post("/api/v1/eventos", response_model=ResultadoIngesta, dependencies=[Depends(autenticar)])
def registrar_eventos(
    eventos: Annotated[list[EventoIn], Body(min_length=1, max_length=500)],
    db: Session = Depends(get_db),
):
    """Ingesta idempotente: un evento_id repetido (reintento del edge) se ignora."""
    stmt = (
        insert(EventoConteo)
        .values([e.model_dump() for e in eventos])
        .on_conflict_do_nothing(index_elements=["evento_id"])
    )
    resultado = db.execute(stmt)
    db.commit()
    return ResultadoIngesta(recibidos=len(eventos), insertados=resultado.rowcount)


@app.get("/api/v1/conteos", response_model=list[ConteoHora], dependencies=[Depends(autenticar)])
def conteos_por_hora(
    dispositivo_id: str | None = None,
    desde: datetime | None = None,
    hasta: datetime | None = None,
    db: Session = Depends(get_db),
):
    """Totales por hora y clase, opcionalmente filtrados por dispositivo y rango de fechas."""
    hora = func.date_trunc("hour", EventoConteo.timestamp).label("hora")
    stmt = select(hora, EventoConteo.clase_objeto, func.count().label("total")).group_by(hora, EventoConteo.clase_objeto)
    if dispositivo_id:
        stmt = stmt.where(EventoConteo.dispositivo_id == dispositivo_id)
    if desde:
        stmt = stmt.where(EventoConteo.timestamp >= desde)
    if hasta:
        stmt = stmt.where(EventoConteo.timestamp < hasta)
    filas = db.execute(stmt.order_by(hora)).all()
    return [ConteoHora(hora=f.hora, clase_objeto=f.clase_objeto, total=f.total) for f in filas]
