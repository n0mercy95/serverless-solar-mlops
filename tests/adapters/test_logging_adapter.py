# ============================================================
# test_logging_adapter.py — Tests del Adaptador de Logging
# Serverless Solar MLOps | Sub-fase 2.2 + 4.2
# ============================================================

import json
import logging
import math
import os
from io import StringIO
from unittest.mock import patch

import pytest

from adapters.logging_adapter import CloudMetricsLoggerAdapter, CloudStructuredLogFormatter


class TestCloudStructuredLogFormatter:
    """Tests del formateador JSON de logs."""

    def setup_method(self):
        self.formatter = CloudStructuredLogFormatter()
        self.record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="fake.py",
            lineno=1,
            msg="Mensaje de prueba",
            args=(),
            exc_info=None,
        )

    def test_basic_formatting(self):
        """Un log básico se formatea como JSON válido."""
        result = self.formatter.format(self.record)
        data = json.loads(result)
        
        assert data["severity"] == "INFO"
        assert data["message"] == "Mensaje de prueba"
        assert data["logger"] == "test_logger"

    def test_extra_json_payload(self):
        """El campo jsonPayload en 'extra' se combina en el JSON raíz."""
        self.record.jsonPayload = {"epoch": 5, "loss": 0.04}
        
        result = self.formatter.format(self.record)
        data = json.loads(result)
        
        assert data["epoch"] == 5
        assert data["loss"] == 0.04
        assert data["severity"] == "INFO"


class TestCloudMetricsLoggerAdapter:
    """Tests del adaptador de métricas."""

    def setup_method(self):
        # Redirigir stdout/stderr del logger a un buffer de memoria
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(CloudStructuredLogFormatter())
        
        self.logger_name = "test_metrics_logger"
        self.adapter = CloudMetricsLoggerAdapter(self.logger_name)
        
        # Sobrescribir el handler interno para test
        self.adapter._logger.handlers = [self.handler]

    def _get_last_log(self) -> dict:
        """Helper para parsear la última línea del log como JSON."""
        output = self.log_stream.getvalue().strip()
        lines = output.split('\n')
        return json.loads(lines[-1])

    def test_log_epoch_metrics(self):
        """log_epoch_metrics emite el JSON correcto."""
        self.adapter.log_epoch_metrics(epoch=10, metrics={"train_loss": 0.1, "val_mae": 0.05})
        
        log_data = self._get_last_log()
        
        assert log_data["severity"] == "INFO"
        assert log_data["event_type"] == "epoch_metrics"
        assert log_data["epoch"] == 10
        assert log_data["train_loss"] == 0.1
        assert log_data["val_mae"] == 0.05

    def test_log_training_complete(self):
        """log_training_complete emite el JSON correcto."""
        self.adapter.log_training_complete(final_metrics={"best_epoch": 42, "best_val_mae": 0.02})
        
        log_data = self._get_last_log()
        
        assert log_data["severity"] == "INFO"
        assert log_data["event_type"] == "training_complete"
        assert log_data["best_epoch"] == 42
        assert log_data["best_val_mae"] == 0.02


class TestCheckDivergence:
    """Tests de detección temprana de divergencia algorítmica (Sub-fase 4.2)."""

    def setup_method(self):
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(CloudStructuredLogFormatter())
        self.handler.setLevel(logging.DEBUG)

        self.adapter = CloudMetricsLoggerAdapter("test_divergence_logger")
        self.adapter._logger.handlers = [self.handler]
        self.adapter._logger.setLevel(logging.DEBUG)

    def _get_last_log(self) -> dict:
        """Helper para parsear la última línea del log como JSON."""
        output = self.log_stream.getvalue().strip()
        lines = output.split('\n')
        return json.loads(lines[-1])

    def test_no_divergence_healthy_metrics(self):
        """Métricas saludables no disparan divergencia."""
        result = self.adapter.check_divergence(
            epoch=5,
            metrics={"train_loss": 0.05, "val_loss": 0.03},
        )
        assert result is False

    def test_divergence_nan_train_loss(self):
        """NaN en train_loss dispara divergencia."""
        result = self.adapter.check_divergence(
            epoch=10,
            metrics={"train_loss": float("nan"), "val_loss": 0.03},
        )
        assert result is True

        log_data = self._get_last_log()
        assert log_data["event_type"] == "divergence_detected"
        assert log_data["divergence_type"] == "nan_or_inf"
        assert log_data["severity"] == "CRITICAL"

    def test_divergence_inf_val_loss(self):
        """Inf en val_loss dispara divergencia."""
        result = self.adapter.check_divergence(
            epoch=15,
            metrics={"train_loss": 0.04, "val_loss": float("inf")},
        )
        assert result is True

        log_data = self._get_last_log()
        assert log_data["divergence_type"] == "nan_or_inf"

    def test_divergence_train_loss_exceeds_threshold(self):
        """train_loss por encima del umbral dispara divergencia."""
        self.adapter._train_loss_threshold = 1.0

        result = self.adapter.check_divergence(
            epoch=20,
            metrics={"train_loss": 1.5, "val_loss": 0.03},
        )
        assert result is True

        log_data = self._get_last_log()
        assert log_data["event_type"] == "divergence_detected"
        assert log_data["divergence_type"] == "train_loss_threshold_exceeded"
        assert log_data["train_loss"] == 1.5
        assert log_data["threshold"] == 1.0

    def test_divergence_val_loss_exceeds_threshold(self):
        """val_loss por encima del umbral dispara divergencia."""
        self.adapter._val_loss_threshold = 0.5

        result = self.adapter.check_divergence(
            epoch=25,
            metrics={"train_loss": 0.04, "val_loss": 0.8},
        )
        assert result is True

        log_data = self._get_last_log()
        assert log_data["divergence_type"] == "val_loss_threshold_exceeded"
        assert log_data["val_loss"] == 0.8
        assert log_data["threshold"] == 0.5

    def test_divergence_thresholds_from_env(self):
        """Los umbrales se cargan correctamente de variables de entorno."""
        with patch.dict(os.environ, {
            "ALERT_TRAIN_LOSS_THRESHOLD": "0.5",
            "ALERT_VAL_LOSS_THRESHOLD": "0.2",
        }):
            adapter = CloudMetricsLoggerAdapter("test_env_thresholds")

            assert adapter._train_loss_threshold == 0.5
            assert adapter._val_loss_threshold == 0.2

    def test_no_divergence_without_loss_keys(self):
        """Sin claves train_loss/val_loss, no hay divergencia por umbrales."""
        result = self.adapter.check_divergence(
            epoch=30,
            metrics={"custom_metric": 0.1},
        )
        assert result is False

    def test_divergence_nan_in_custom_metric(self):
        """NaN en cualquier métrica (no solo loss) dispara divergencia."""
        result = self.adapter.check_divergence(
            epoch=35,
            metrics={"custom_metric": float("nan")},
        )
        assert result is True


class TestLogInferenceMetrics:
    """Tests del logging de métricas de inferencia (Sub-fase 4.2)."""

    def setup_method(self):
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(CloudStructuredLogFormatter())

        self.adapter = CloudMetricsLoggerAdapter("test_inference_logger")
        self.adapter._logger.handlers = [self.handler]

    def _get_last_log(self) -> dict:
        output = self.log_stream.getvalue().strip()
        lines = output.split('\n')
        return json.loads(lines[-1])

    def test_log_inference_metrics_basic(self):
        """log_inference_metrics emite el JSON correcto."""
        self.adapter.log_inference_metrics(
            metrics={"latency_ms": 45.2, "prediction_mae": 0.03},
            model_version="v2",
        )

        log_data = self._get_last_log()

        assert log_data["severity"] == "INFO"
        assert log_data["event_type"] == "inference_metrics"
        assert log_data["model_version"] == "v2"
        assert log_data["latency_ms"] == 45.2
        assert log_data["prediction_mae"] == 0.03

    def test_log_inference_metrics_default_version(self):
        """Sin model_version, usa 'unknown' por defecto."""
        self.adapter.log_inference_metrics(
            metrics={"latency_ms": 50.0},
        )

        log_data = self._get_last_log()
        assert log_data["model_version"] == "unknown"

