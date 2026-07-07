# ============================================================
# test_monitoring.py — Tests del Módulo de Monitorización
# Serverless Solar MLOps | Sub-fase 4.2
# ============================================================
# Valida la definición de métricas, umbrales de alerta,
# el provisioning con mocking del SDK de GCP, y la exportación
# de configuración como JSON.
# ============================================================

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adapters.monitoring import (
    AlertThreshold,
    MetricDefinition,
    export_monitoring_config,
    get_default_alert_thresholds,
    get_default_metric_definitions,
    provision_alert_policies,
    provision_log_based_metrics,
)


class TestMetricDefinition:
    """Tests para la configuración de métricas."""

    def test_default_metrics_count(self) -> None:
        """Se generan exactamente 3 métricas predeterminadas."""
        metrics = get_default_metric_definitions("test-project")
        assert len(metrics) == 3

    def test_default_metrics_names(self) -> None:
        """Las métricas tienen los nombres esperados."""
        metrics = get_default_metric_definitions("test-project")
        names = [m.name for m in metrics]

        assert "solar_mlops/epoch_train_loss" in names
        assert "solar_mlops/epoch_val_loss" in names
        assert "solar_mlops/training_final_val_loss" in names

    def test_metric_filter_contains_event_type(self) -> None:
        """Cada filtro de métrica incluye el event_type correcto."""
        metrics = get_default_metric_definitions("test-project")

        for metric in metrics:
            assert "jsonPayload.event_type=" in metric.filter_query

    def test_metric_value_extractors(self) -> None:
        """Cada métrica tiene un value_extractor válido."""
        metrics = get_default_metric_definitions("test-project")

        for metric in metrics:
            assert metric.value_extractor.startswith("EXTRACT(")
            assert metric.value_extractor.endswith(")")

    def test_metric_default_types(self) -> None:
        """Las métricas tienen los tipos por defecto correctos."""
        metrics = get_default_metric_definitions("test-project")

        for metric in metrics:
            assert metric.metric_kind == "GAUGE"
            assert metric.value_type == "DOUBLE"


class TestAlertThreshold:
    """Tests para la configuración de alertas."""

    def test_default_alerts_count(self) -> None:
        """Se generan exactamente 3 alertas predeterminadas."""
        alerts = get_default_alert_thresholds()
        assert len(alerts) == 3

    def test_default_alert_thresholds_values(self) -> None:
        """Los umbrales predeterminados tienen los valores esperados."""
        alerts = get_default_alert_thresholds()

        train_loss_alert = next(
            a for a in alerts if "train_loss" in a.metric_name
        )
        val_loss_alert = next(
            a for a in alerts if "epoch_val_loss" in a.metric_name
        )
        final_alert = next(
            a for a in alerts if "final_val_loss" in a.metric_name
        )

        assert train_loss_alert.threshold_value == 1.0
        assert val_loss_alert.threshold_value == 0.5
        assert final_alert.threshold_value == 0.3

    def test_alert_thresholds_configurable_via_env(self) -> None:
        """Los umbrales pueden configurarse mediante variables de entorno."""
        with patch.dict(os.environ, {
            "ALERT_TRAIN_LOSS_THRESHOLD": "2.5",
            "ALERT_VAL_LOSS_THRESHOLD": "0.8",
            "ALERT_FINAL_VAL_LOSS_THRESHOLD": "0.6",
        }):
            alerts = get_default_alert_thresholds()

            train_alert = next(
                a for a in alerts if "train_loss" in a.metric_name
            )
            val_alert = next(
                a for a in alerts if "epoch_val_loss" in a.metric_name
            )
            final_alert = next(
                a for a in alerts if "final_val_loss" in a.metric_name
            )

            assert train_alert.threshold_value == 2.5
            assert val_alert.threshold_value == 0.8
            assert final_alert.threshold_value == 0.6

    def test_alert_notification_channels(self) -> None:
        """Los canales de notificación se propagan correctamente."""
        channels = ["projects/p/notificationChannels/123"]
        alerts = get_default_alert_thresholds(notification_channels=channels)

        for alert in alerts:
            assert alert.notification_channels == channels

    def test_alert_comparison_type(self) -> None:
        """Todas las alertas usan comparación COMPARISON_GT."""
        alerts = get_default_alert_thresholds()

        for alert in alerts:
            assert alert.comparison == "COMPARISON_GT"


class TestProvisionLogBasedMetrics:
    """Tests para la función de provisioning de métricas."""

    def test_provision_local_fallback_without_gcp(self) -> None:
        """Sin google-cloud-logging, retorna definiciones locales."""
        # Forzar que el import falle independientemente de si está instalado
        import sys
        with patch.dict(sys.modules, {"google.cloud.logging": None, "google.cloud": None, "google": None}):
            # Reimportar para que use el fallback
            import importlib
            import adapters.monitoring as mon_module
            importlib.reload(mon_module)
            results = mon_module.provision_log_based_metrics("test-project")
            # Restaurar
            importlib.reload(mon_module)

        assert len(results) == 3
        for result in results:
            assert result["status"] == "local_definition"
            assert "filter_query" in result
            assert "value_extractor" in result

    def test_provision_with_mock_client(self) -> None:
        """Con client mock, las métricas se crean exitosamente."""
        mock_client = MagicMock()
        mock_cloud_logging = MagicMock()
        mock_cloud_logging.Client.return_value = mock_client
        mock_client.metrics_api.metric_create.return_value = MagicMock()

        import sys
        with patch.dict(sys.modules, {"google.cloud.logging": mock_cloud_logging, "google.cloud": MagicMock(), "google": MagicMock()}):
            results = provision_log_based_metrics("test-project")
            assert len(results) == 3


class TestProvisionAlertPolicies:
    """Tests para la función de provisioning de alertas."""

    def test_provision_local_fallback_without_gcp(self) -> None:
        """Sin google-cloud-monitoring, retorna definiciones locales."""
        import sys
        with patch.dict(sys.modules, {"google.cloud.monitoring_v3": None, "google.cloud": None, "google": None}):
            import importlib
            import adapters.monitoring as mon_module
            importlib.reload(mon_module)
            results = mon_module.provision_alert_policies("test-project")
            importlib.reload(mon_module)

        assert len(results) == 3
        for result in results:
            assert result["status"] == "local_definition"
            assert "threshold" in result
            assert "description" in result



class TestExportMonitoringConfig:
    """Tests para la exportación de configuración."""

    def test_export_returns_complete_config(self) -> None:
        """La exportación retorna un diccionario completo."""
        config = export_monitoring_config("test-project")

        assert config["project_id"] == "test-project"
        assert len(config["log_based_metrics"]) == 3
        assert len(config["alert_policies"]) == 3

    def test_export_metrics_structure(self) -> None:
        """Cada métrica exportada tiene todos los campos requeridos."""
        config = export_monitoring_config("test-project")

        for metric in config["log_based_metrics"]:
            assert "name" in metric
            assert "display_name" in metric
            assert "filter_query" in metric
            assert "value_extractor" in metric
            assert "metric_kind" in metric
            assert "value_type" in metric

    def test_export_alerts_structure(self) -> None:
        """Cada alerta exportada tiene todos los campos requeridos."""
        config = export_monitoring_config("test-project")

        for alert in config["alert_policies"]:
            assert "metric_name" in alert
            assert "display_name" in alert
            assert "threshold_value" in alert
            assert "comparison" in alert

    def test_export_to_file(self, tmp_path: Path) -> None:
        """La exportación a archivo genera un JSON válido."""
        output_file = tmp_path / "monitoring_config.json"

        config = export_monitoring_config(
            "test-project",
            output_path=str(output_file),
        )

        assert output_file.exists()
        with open(output_file, "r") as f:
            saved_config = json.load(f)

        assert saved_config["project_id"] == "test-project"
        assert len(saved_config["log_based_metrics"]) == 3
        assert len(saved_config["alert_policies"]) == 3

    def test_export_config_is_json_serializable(self) -> None:
        """La configuración exportada es completamente serializable a JSON."""
        config = export_monitoring_config("test-project")
        # No debe lanzar excepción
        json_str = json.dumps(config)
        assert len(json_str) > 0
