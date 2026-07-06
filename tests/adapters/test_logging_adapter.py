# ============================================================
# test_logging_adapter.py — Tests del Adaptador de Logging
# Serverless Solar MLOps | Sub-fase 2.2
# ============================================================

import json
import logging
from io import StringIO

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
