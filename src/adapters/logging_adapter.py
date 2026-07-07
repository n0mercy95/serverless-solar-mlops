# ============================================================
# logging_adapter.py — Adaptador de Logging Estructurado
# Serverless Solar MLOps | Sub-fase 2.2 + 4.2 (Monitoring)
# ============================================================
# Implementación de MetricsLoggerPort utilizando la librería
# estándar de logging de Python, pero formateando la salida
# a JSON. Esto permite que el agente de Google Cloud Logging
# parsee automáticamente el `jsonPayload` para facilitar
# Log-based Metrics en Cloud Monitoring.
#
# Sub-fase 4.2: Se añade detección temprana de divergencia
# algorítmica y logging de métricas de inferencia.
# ============================================================

import json
import logging
import math
import os
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

    Sub-fase 4.2: Incluye detección de divergencia algorítmica
    temprana y logging de métricas de inferencia.
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

        # Umbrales de divergencia configurables vía variables de entorno
        self._train_loss_threshold = float(
            os.environ.get("ALERT_TRAIN_LOSS_THRESHOLD", "1.0")
        )
        self._val_loss_threshold = float(
            os.environ.get("ALERT_VAL_LOSS_THRESHOLD", "0.5")
        )

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

    def check_divergence(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Verifica si las métricas de la época indican divergencia algorítmica.

        Detecta tres tipos de divergencia:
        1. NaN/Inf en cualquier métrica (gradientes explosivos).
        2. train_loss supera el umbral de divergencia.
        3. val_loss supera el umbral de calidad mínima.

        Args:
            epoch: Número de la época actual.
            metrics: Diccionario con las métricas de la época.

        Returns:
            True si se detecta divergencia, False si el entrenamiento es saludable.
        """
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        # Detección de NaN / Inf
        for key, value in metrics.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                self._logger.critical(
                    f"DIVERGENCIA DETECTADA: {key} = {value} en época {epoch}",
                    extra={"jsonPayload": {
                        "event_type": "divergence_detected",
                        "divergence_type": "nan_or_inf",
                        "metric_name": key,
                        "metric_value": str(value),
                        "epoch": epoch,
                        "severity": "CRITICAL",
                    }}
                )
                return True

        # Detección de train_loss por encima del umbral
        if train_loss is not None and train_loss > self._train_loss_threshold:
            self._logger.warning(
                f"ALERTA: train_loss ({train_loss:.6f}) supera el umbral "
                f"({self._train_loss_threshold}) en época {epoch}",
                extra={"jsonPayload": {
                    "event_type": "divergence_detected",
                    "divergence_type": "train_loss_threshold_exceeded",
                    "train_loss": train_loss,
                    "threshold": self._train_loss_threshold,
                    "epoch": epoch,
                    "severity": "WARNING",
                }}
            )
            return True

        # Detección de val_loss por encima del umbral
        if val_loss is not None and val_loss > self._val_loss_threshold:
            self._logger.warning(
                f"ALERTA: val_loss ({val_loss:.6f}) supera el umbral "
                f"({self._val_loss_threshold}) en época {epoch}",
                extra={"jsonPayload": {
                    "event_type": "divergence_detected",
                    "divergence_type": "val_loss_threshold_exceeded",
                    "val_loss": val_loss,
                    "threshold": self._val_loss_threshold,
                    "epoch": epoch,
                    "severity": "WARNING",
                }}
            )
            return True

        return False

    def log_inference_metrics(
        self, metrics: Dict[str, float], model_version: str = "unknown"
    ) -> None:
        """Registra métricas de inferencia para monitorización en producción.

        Emite un log JSON estructurado que Cloud Monitoring puede capturar
        para detectar degradación del modelo en tiempo de inferencia.

        Args:
            metrics: Diccionario con métricas de inferencia
                (ej. {"latency_ms": 45.2, "prediction_mae": 0.03}).
            model_version: Versión del modelo que generó la predicción.
        """
        payload = {
            "event_type": "inference_metrics",
            "model_version": model_version,
        }
        payload.update(metrics)

        self._logger.info(
            "Métricas de inferencia registradas",
            extra={"jsonPayload": payload}
        )

