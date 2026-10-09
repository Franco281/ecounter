-- Referencia del esquema (lo crea app.init_db via SQLAlchemy).
CREATE TABLE IF NOT EXISTS eventos_conteo (
    id             BIGSERIAL PRIMARY KEY,
    evento_id      UUID NOT NULL UNIQUE,
    dispositivo_id VARCHAR(64) NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL,
    clase_objeto   VARCHAR(32) NOT NULL,
    direccion      VARCHAR(16) NOT NULL,
    confianza      DOUBLE PRECISION NOT NULL,
    recibido_en    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_eventos_dispositivo_timestamp ON eventos_conteo (dispositivo_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_eventos_timestamp ON eventos_conteo (timestamp);
