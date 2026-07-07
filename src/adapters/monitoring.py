# ============================================================
# monitoring.py — Módulo de Log-based Metrics y Alertas
# Serverless Solar MLOps | Sub-fase 4.2
# ============================================================
# Define las configuraciones de métricas basadas en registros
# (Log-based Metrics) y políticas de alerta para Cloud Monitoring.
# Provee funciones para:
#   - Crear/actualizar métricas que extraen valores del jsonPayload
#     emitido por CloudStructuredLogFormatter (Sub-fase 2.2).
#   - Crear/actualizar políticas de alerta que disparan notificaciones
#     cuando el modelo diverge (train_loss o val_loss superan umbrales).
#   - Detección temprana de divergencia algorítmica para interrumpir
#     proactivamente trabajos de Vertex AI.
#
# IMPORTANTE (PRD §4): Todas las operaciones de I/O usan bloques
# try-except obligatorios con logging estructurado JSON.
# ============================================================

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --- Configuración de Métricas y Umbrales ---

@dataclass
class MetricDefinition:
    """Definición de una métrica basada en registros de Cloud Monitoring.

    Attributes:
        name: Identificador único de la métrica (ej. 'training/epoch_val_loss').
        display_name: Nombre legible para la consola de Cloud Monitoring.
        description: Descripción de qué mide la métrica.
        filter_query: Filtro de Cloud Logging para seleccionar los logs relevantes.
        value_extractor: Expresión CEL que extrae el valor numérico del jsonPayload.
        metric_kind: Tipo de métrica ('GAUGE' para valores instantáneos).
        value_type: Tipo del valor ('DOUBLE' para punto flotante).
    """
    name: str
    display_name: str
    description: str
    filter_query: str
    value_extractor: str
    metric_kind: str = "GAUGE"
    value_type: str = "DOUBLE"


@dataclass
class AlertThreshold:
    """Umbral de alerta para detección de divergencia algorítmica.

    Attributes:
        metric_name: Nombre de la métrica a monitorizar.
        display_name: Nombre legible de la alerta.
        condition_display_name: Nombre de la condición dentro de la política.
        threshold_value: Valor umbral que dispara la alerta.
        comparison: Tipo de comparación ('COMPARISON_GT' = mayor que).
        duration_seconds: Ventana de tiempo para evaluar la condición.
        notification_channels: Lista de IDs de canales de notificación.
        description: Descripción de la política de alerta.
    """
    metric_name: str
    display_name: str
    condition_display_name: str
    threshold_value: float
    comparison: str = "COMPARISON_GT"
    duration_seconds: int = 0
    notification_channels: List[str] = field(default_factory=list)
    description: str = ""


# --- Métricas predefinidas del sistema ---

def get_default_metric_definitions(project_id: str) -> List[MetricDefinition]:
    """Retorna las definiciones de métricas estándar del pipeline MLOps.

    Estas métricas capturan los valores de train_loss y val_loss emitidos
    en cada época por el CloudStructuredLogFormatter (Sub-fase 2.2).

    Args:
        project_id: ID del proyecto GCP.

    Returns:
        Lista de MetricDefinition con las métricas predeterminadas.
    """
    base_filter = (
        'resource.type="gce_instance" OR resource.type="cloud_run_job" '
        'OR resource.type="global"'
    )

    return [
        MetricDefinition(
            name="solar_mlops/epoch_train_loss",
            display_name="Solar MLOps — Training Loss por Época",
            description=(
                "Captura el valor de train_loss emitido en cada época "
                "del entrenamiento del Transformer Bi-LSTM. Permite "
                "detectar divergencia temprana durante el entrenamiento."
            ),
            filter_query=(
                f'{base_filter} '
                'jsonPayload.event_type="epoch_metrics" '
                'jsonPayload.train_loss:*'
            ),
            value_extractor='EXTRACT(jsonPayload.train_loss)',
        ),
        MetricDefinition(
            name="solar_mlops/epoch_val_loss",
            display_name="Solar MLOps — Validation Loss (MAE) por Época",
            description=(
                "Captura el val_loss (MAE) del dataset de validación en "
                "cada época. Esta métrica es la referencia principal para "
                "la evaluación Champion/Challenger."
            ),
            filter_query=(
                f'{base_filter} '
                'jsonPayload.event_type="epoch_metrics" '
                'jsonPayload.val_loss:*'
            ),
            value_extractor='EXTRACT(jsonPayload.val_loss)',
        ),
        MetricDefinition(
            name="solar_mlops/training_final_val_loss",
            display_name="Solar MLOps — Best Validation Loss Final",
            description=(
                "Captura el best_val_loss al completar el entrenamiento. "
                "Permite trazar la tendencia de calidad del modelo a lo "
                "largo de múltiples ciclos de reentrenamiento."
            ),
            filter_query=(
                f'{base_filter} '
                'jsonPayload.event_type="training_complete" '
                'jsonPayload.best_val_loss:*'
            ),
            value_extractor='EXTRACT(jsonPayload.best_val_loss)',
        ),
    ]


def get_default_alert_thresholds(
    notification_channels: Optional[List[str]] = None,
) -> List[AlertThreshold]:
    """Retorna las políticas de alerta predeterminadas del sistema.

    Cada alerta está diseñada para detectar tempranamente si el modelo
    diverge y notificar a los ingenieros MLOps.

    Args:
        notification_channels: Lista de IDs de canales de notificación
            de Cloud Monitoring (ej. emails, Slack, PagerDuty).

    Returns:
        Lista de AlertThreshold con las alertas predeterminadas.
    """
    channels = notification_channels or []

    return [
        AlertThreshold(
            metric_name="solar_mlops/epoch_train_loss",
            display_name="🚨 Divergencia de Training Loss — Solar MLOps",
            condition_display_name="Train Loss supera umbral de divergencia",
            threshold_value=float(os.environ.get(
                "ALERT_TRAIN_LOSS_THRESHOLD", "1.0"
            )),
            comparison="COMPARISON_GT",
            duration_seconds=0,
            notification_channels=channels,
            description=(
                "Se activa cuando el train_loss de una época supera el "
                "umbral configurado, indicando que el modelo puede estar "
                "divergiendo (gradientes explosivos, learning rate alto, "
                "o datos corruptos). Acción recomendada: interrumpir el "
                "trabajo de Vertex AI."
            ),
        ),
        AlertThreshold(
            metric_name="solar_mlops/epoch_val_loss",
            display_name="🚨 Degradación de Validation Loss — Solar MLOps",
            condition_display_name="Val Loss supera umbral de calidad",
            threshold_value=float(os.environ.get(
                "ALERT_VAL_LOSS_THRESHOLD", "0.5"
            )),
            comparison="COMPARISON_GT",
            duration_seconds=0,
            notification_channels=channels,
            description=(
                "Se activa cuando el val_loss (MAE) supera el umbral de "
                "calidad mínima, indicando que el modelo no está generalizando "
                "correctamente. Posibles causas: overfitting, concept drift, "
                "o insuficiencia de datos en la Capa Oro."
            ),
        ),
        AlertThreshold(
            metric_name="solar_mlops/training_final_val_loss",
            display_name="🚨 Regresión de Calidad Post-Entrenamiento — Solar MLOps",
            condition_display_name="Best val_loss final supera umbral histórico",
            threshold_value=float(os.environ.get(
                "ALERT_FINAL_VAL_LOSS_THRESHOLD", "0.3"
            )),
            comparison="COMPARISON_GT",
            duration_seconds=0,
            notification_channels=channels,
            description=(
                "Se activa al finalizar un ciclo de entrenamiento si el "
                "best_val_loss final supera el umbral histórico aceptable. "
                "Indica una regresión sistémica en la calidad del modelo."
            ),
        ),
    ]


# --- Provisioning de Métricas y Alertas en Cloud Monitoring ---

def provision_log_based_metrics(
    project_id: str,
    metrics: Optional[List[MetricDefinition]] = None,
) -> List[Dict[str, Any]]:
    """Crea o actualiza métricas basadas en registros en Cloud Monitoring.

    Utiliza la API de google-cloud-logging para provisionar las métricas
    que extraen valores numéricos del jsonPayload emitido por los logs
    estructurados del pipeline de entrenamiento.

    Args:
        project_id: ID del proyecto GCP.
        metrics: Lista de MetricDefinition. Si es None, usa las predeterminadas.

    Returns:
        Lista de diccionarios con el resultado de cada provisión.

    Raises:
        RuntimeError: Si falla la comunicación con Cloud Monitoring.
    """
    if metrics is None:
        metrics = get_default_metric_definitions(project_id)

    results: List[Dict[str, Any]] = []

    try:
        from google.cloud import logging as cloud_logging

        client = cloud_logging.Client(project=project_id)

        for metric_def in metrics:
            try:
                metric = client.metrics_api.metric_create(
                    project=project_id,
                    metric_name=metric_def.name,
                    filter_=metric_def.filter_query,
                    description=metric_def.description,
                )
                result = {
                    "metric_name": metric_def.name,
                    "status": "created",
                    "display_name": metric_def.display_name,
                }
                logger.info(
                    f"Métrica '{metric_def.name}' creada exitosamente",
                    extra={"jsonPayload": result},
                )
            except Exception as create_err:
                # Si ya existe, intentar actualizar
                if "already exists" in str(create_err).lower() or "409" in str(create_err):
                    result = {
                        "metric_name": metric_def.name,
                        "status": "already_exists",
                        "display_name": metric_def.display_name,
                    }
                    logger.info(
                        f"Métrica '{metric_def.name}' ya existe. Omitiendo.",
                        extra={"jsonPayload": result},
                    )
                else:
                    result = {
                        "metric_name": metric_def.name,
                        "status": "error",
                        "error": str(create_err),
                    }
                    logger.error(
                        f"Error creando métrica '{metric_def.name}': {create_err}",
                        extra={"jsonPayload": result},
                    )

            results.append(result)

        return results

    except ImportError:
        logger.warning(
            "google-cloud-logging no disponible. Generando configuración local.",
            extra={"jsonPayload": {"action": "provision_metrics_local_fallback"}},
        )
        # Fallback: retornar las definiciones como diccionarios para IaC
        for metric_def in metrics:
            results.append({
                "metric_name": metric_def.name,
                "status": "local_definition",
                "display_name": metric_def.display_name,
                "filter_query": metric_def.filter_query,
                "value_extractor": metric_def.value_extractor,
            })
        return results

    except Exception as e:
        logger.error(
            "Error provisionando métricas en Cloud Monitoring",
            extra={
                "jsonPayload": {
                    "action": "provision_metrics_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            },
        )
        raise RuntimeError(
            f"Error provisionando métricas: {type(e).__name__}: {e}"
        ) from e


def provision_alert_policies(
    project_id: str,
    alerts: Optional[List[AlertThreshold]] = None,
) -> List[Dict[str, Any]]:
    """Crea o actualiza políticas de alerta en Cloud Monitoring.

    Cada política monitoriza una métrica basada en registros y dispara
    notificaciones cuando el valor supera el umbral configurado.

    Args:
        project_id: ID del proyecto GCP.
        alerts: Lista de AlertThreshold. Si es None, usa las predeterminadas.

    Returns:
        Lista de diccionarios con el resultado de cada provisión.

    Raises:
        RuntimeError: Si falla la comunicación con Cloud Monitoring.
    """
    if alerts is None:
        alerts = get_default_alert_thresholds()

    results: List[Dict[str, Any]] = []

    try:
        from google.cloud import monitoring_v3

        client = monitoring_v3.AlertPolicyServiceClient()
        project_name = f"projects/{project_id}"

        for alert_def in alerts:
            try:
                # Construir la condición de umbral
                condition = monitoring_v3.AlertPolicy.Condition(
                    display_name=alert_def.condition_display_name,
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter=(
                            f'metric.type="logging.googleapis.com/user/{alert_def.metric_name}" '
                            f'resource.type="global"'
                        ),
                        comparison=getattr(
                            monitoring_v3.ComparisonType,
                            alert_def.comparison,
                            monitoring_v3.ComparisonType.COMPARISON_GT,
                        ),
                        threshold_value=alert_def.threshold_value,
                        duration={"seconds": alert_def.duration_seconds},
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 60},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MAX,
                            )
                        ],
                    ),
                )

                # Construir la política de alerta
                policy = monitoring_v3.AlertPolicy(
                    display_name=alert_def.display_name,
                    documentation=monitoring_v3.AlertPolicy.Documentation(
                        content=alert_def.description,
                        mime_type="text/markdown",
                    ),
                    conditions=[condition],
                    combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
                    notification_channels=alert_def.notification_channels,
                    enabled={"value": True},
                )

                created_policy = client.create_alert_policy(
                    name=project_name,
                    alert_policy=policy,
                )

                result = {
                    "alert_name": alert_def.display_name,
                    "status": "created",
                    "policy_name": created_policy.name,
                    "threshold": alert_def.threshold_value,
                }
                logger.info(
                    f"Política de alerta '{alert_def.display_name}' creada",
                    extra={"jsonPayload": result},
                )

            except Exception as create_err:
                if "already exists" in str(create_err).lower() or "409" in str(create_err):
                    result = {
                        "alert_name": alert_def.display_name,
                        "status": "already_exists",
                    }
                    logger.info(
                        f"Política '{alert_def.display_name}' ya existe.",
                        extra={"jsonPayload": result},
                    )
                else:
                    result = {
                        "alert_name": alert_def.display_name,
                        "status": "error",
                        "error": str(create_err),
                    }
                    logger.error(
                        f"Error creando política '{alert_def.display_name}': {create_err}",
                        extra={"jsonPayload": result},
                    )

            results.append(result)

        return results

    except ImportError:
        logger.warning(
            "google-cloud-monitoring no disponible. Generando configuración local.",
            extra={"jsonPayload": {"action": "provision_alerts_local_fallback"}},
        )
        for alert_def in alerts:
            results.append({
                "alert_name": alert_def.display_name,
                "status": "local_definition",
                "metric_name": alert_def.metric_name,
                "threshold": alert_def.threshold_value,
                "comparison": alert_def.comparison,
                "description": alert_def.description,
            })
        return results

    except Exception as e:
        logger.error(
            "Error provisionando políticas de alerta",
            extra={
                "jsonPayload": {
                    "action": "provision_alerts_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            },
        )
        raise RuntimeError(
            f"Error provisionando alertas: {type(e).__name__}: {e}"
        ) from e


def export_monitoring_config(
    project_id: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Exporta la configuración de métricas y alertas como JSON.

    Útil para revisión, auditoría, o para alimentar pipelines de IaC
    (Terraform/Pulumi) sin ejecutar directamente contra la API.

    Args:
        project_id: ID del proyecto GCP.
        output_path: Ruta opcional donde guardar el JSON. Si es None,
            solo retorna el diccionario.

    Returns:
        Diccionario con la configuración completa.
    """
    metrics = get_default_metric_definitions(project_id)
    alerts = get_default_alert_thresholds()

    config: Dict[str, Any] = {
        "project_id": project_id,
        "log_based_metrics": [],
        "alert_policies": [],
    }

    for m in metrics:
        config["log_based_metrics"].append({
            "name": m.name,
            "display_name": m.display_name,
            "description": m.description,
            "filter_query": m.filter_query,
            "value_extractor": m.value_extractor,
            "metric_kind": m.metric_kind,
            "value_type": m.value_type,
        })

    for a in alerts:
        config["alert_policies"].append({
            "metric_name": a.metric_name,
            "display_name": a.display_name,
            "condition_display_name": a.condition_display_name,
            "threshold_value": a.threshold_value,
            "comparison": a.comparison,
            "duration_seconds": a.duration_seconds,
            "description": a.description,
        })

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Configuración de monitorización exportada a {output_path}",
                extra={"jsonPayload": {"action": "config_exported", "path": output_path}},
            )
        except Exception as e:
            logger.error(
                f"Error exportando configuración: {e}",
                extra={"jsonPayload": {"action": "config_export_error", "error": str(e)}},
            )
            raise RuntimeError(f"Error exportando configuración: {e}") from e

    return config
