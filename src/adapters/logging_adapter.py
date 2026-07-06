# ============================================================
# logging_adapter.py — Adaptador de Logging Estructurado
# Serverless Solar MLOps | Sub-fase 2.2
# ============================================================
# Implementación de MetricsLoggerPort utilizando la librería
# estándar de logging de Python, pero formateando la salida
# a JSON. Esto permite que el agente de Google Cloud Logging
# parsee automáticamente el `jsonPayload` para facilitar
# Log-based Metrics en Cloud Monitoring.
# ============================================================

import json
import logging
from typing import Any, Dict

from domain.ports.ports import MetricsLoggerPort
from adapters.model_adapters import _make_json_serializable


class CloudStructuredLogFormatter(logging.Formatter):
    """Formateador personalizado que emite todos los logs como JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Formatea el registro de log como un string JSON.
        
        Soporta diccionarios extra pasados en el parámetro 'extra'
        bajo la clave 'jsonPayload'.
        """
        payload: Dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Extraer información de error si existe
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Merge de 'jsonPayload' si fue proveído en el argumento 'extra'
        # ej: logger.info("...", extra={"jsonPayload": {"loss": 0.5}})
        if hasattr(record, "jsonPayload") and isinstance(record.jsonPayload, dict):
            # Aseguramos que los valores sean serializables
            safe_extra = _make_json_serializable(record.jsonPayload)
            payload.update(safe_extra) # type: ignore

        return json.dumps(payload, ensure_ascii=False)


class CloudMetricsLoggerAdapter(MetricsLoggerPort):
    """Adaptador de observabilidad para Vertex AI y Cloud Logging.

    Cumple el contrato MetricsLoggerPort. Emite métricas de
    entrenamiento por época y métricas finales en formato JSON estructurado
    para que puedan ser ingeridas por Cloud Monitoring (Log-based metrics).
    """

    def __init__(self, logger_name: str = "solar_mlops_training") -> None:
        """Inicializa el adaptador y configura el logger subyacente.
        
        Args:
            logger_name: Nombre del logger (default "solar_mlops_training").
        """
        self._logger = logging.getLogger(logger_name)
        
        # Evitar agregar múltiples handlers si se instancia varias veces
        if not self._logger.handlers:
            self._logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(CloudStructuredLogFormatter())
            self._logger.addHandler(handler)
            self._logger.propagate = False

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Registra las métricas de una época completada.

        Args:
            epoch: Número de la época actual.
            metrics: Diccionario con métricas (ej. loss, val_mae).
        """
        payload = {
            "event_type": "epoch_metrics",
            "epoch": epoch,
        }
        payload.update(metrics)

        self._logger.info(
            f"Métricas época {epoch} completada",
            extra={"jsonPayload": payload}
        )

    def log_training_complete(self, final_metrics: Dict[str, float]) -> None:
        """Registra la finalización exitosa del entrenamiento.

        Args:
            final_metrics: Diccionario con las métricas finales.
        """
        payload = {
            "event_type": "training_complete",
        }
        payload.update(final_metrics)

        self._logger.info(
            "Entrenamiento completado exitosamente",
            extra={"jsonPayload": payload}
        )
