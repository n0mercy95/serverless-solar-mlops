# Avance Sub-fase 4.2: Log-based Metrics y Alertas Automáticas

**Rama:** `feat/4.2-monitoring-alerts`
**Fecha:** 2026-07-07
**Estado:** ✅ Completada

---

## Objetivos (PRD §4, línea 61)

> "Activar las métricas en Cloud Monitoring que analicen el jsonPayload extraído durante
> la inferencia y entrenamiento para disparar alertas a los ingenieros MLOps en caso de
> una divergencia algorítmica temprana."

## Entregables

### 1. Módulo de Monitorización (`src/adapters/monitoring.py`)

Nuevo módulo que define y provisiona las métricas basadas en registros (Log-based Metrics)
y las políticas de alerta en Google Cloud Monitoring.

**Métricas definidas:**

| Métrica | Fuente del jsonPayload | Propósito |
|---|---|---|
| `solar_mlops/epoch_train_loss` | `event_type=epoch_metrics` → `train_loss` | Detectar divergencia durante entrenamiento |
| `solar_mlops/epoch_val_loss` | `event_type=epoch_metrics` → `val_loss` | Detectar degradación en validación |
| `solar_mlops/training_final_val_loss` | `event_type=training_complete` → `best_val_loss` | Rastrear calidad inter-ciclos |

**Políticas de alerta:**

| Alerta | Umbral Default | Severidad |
|---|---|---|
| Divergencia de Training Loss | `train_loss > 1.0` | 🚨 WARNING |
| Degradación de Validation Loss | `val_loss > 0.5` | 🚨 WARNING |
| Regresión Post-Entrenamiento | `best_val_loss > 0.3` | 🚨 WARNING |

Todos los umbrales son configurables vía variables de entorno (`ALERT_*_THRESHOLD`).

**Funcionalidades adicionales:**
- `provision_log_based_metrics()`: Crea métricas en Cloud Monitoring vía API.
- `provision_alert_policies()`: Crea políticas de alerta con notificaciones.
- `export_monitoring_config()`: Exporta la configuración como JSON para IaC (Terraform/Pulumi).
- Fallback local cuando los SDKs de GCP no están disponibles.

### 2. Detección de Divergencia en `logging_adapter.py`

Se amplió `CloudMetricsLoggerAdapter` con:

- **`check_divergence(epoch, metrics)`**: Detecta tres tipos de divergencia algorítmica:
  1. **NaN/Inf** en cualquier métrica (gradientes explosivos).
  2. **train_loss** por encima del umbral configurable.
  3. **val_loss** por encima del umbral de calidad mínima.

- **`log_inference_metrics(metrics, model_version)`**: Emite logs JSON para
  monitorización de métricas de inferencia en producción.

### 3. Variables de Entorno (`.env.example`)

Se añadieron las siguientes variables:

```env
ALERT_TRAIN_LOSS_THRESHOLD=1.0
ALERT_VAL_LOSS_THRESHOLD=0.5
ALERT_FINAL_VAL_LOSS_THRESHOLD=0.3
ALERT_NOTIFICATION_CHANNELS=
```

### 4. Tests Unitarios

- **`tests/adapters/test_monitoring.py`** (18 tests): Métricas, alertas, provisioning
  con mocking de GCP, exportación JSON.
- **`tests/adapters/test_logging_adapter.py`** (14 tests): Tests existentes +
  8 nuevos tests para `check_divergence` y `log_inference_metrics`.

**Resultado: 32/32 tests pasando ✅**

## Flujo de Datos

```
 [Entrenamiento Vertex AI]
         │
         ▼
 CloudMetricsLoggerAdapter
   ├── log_epoch_metrics()     → jsonPayload.event_type="epoch_metrics"
   ├── check_divergence()      → jsonPayload.event_type="divergence_detected"
   ├── log_training_complete() → jsonPayload.event_type="training_complete"
   └── log_inference_metrics() → jsonPayload.event_type="inference_metrics"
         │
         ▼
 Google Cloud Logging (jsonPayload parseado automáticamente)
         │
         ▼
 Log-based Metrics (Cloud Monitoring)
   ├── solar_mlops/epoch_train_loss
   ├── solar_mlops/epoch_val_loss
   └── solar_mlops/training_final_val_loss
         │
         ▼
 Alert Policies → Notification Channels (Email, Slack, PagerDuty)
```

## Archivos Modificados / Creados

| Archivo | Acción |
|---|---|
| `src/adapters/monitoring.py` | **NUEVO** — Módulo de Log-based Metrics y Alertas |
| `src/adapters/logging_adapter.py` | **MODIFICADO** — Añadidos `check_divergence()` y `log_inference_metrics()` |
| `.env.example` | **MODIFICADO** — Añadidas variables de umbrales de alerta |
| `tests/adapters/test_monitoring.py` | **NUEVO** — 18 tests del módulo de monitorización |
| `tests/adapters/test_logging_adapter.py` | **MODIFICADO** — 8 tests nuevos para divergencia e inferencia |
| `docs/advances/advance-4.2.md` | **NUEVO** — Este documento |
