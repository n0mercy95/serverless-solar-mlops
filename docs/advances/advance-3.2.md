# 📊 Avance Sub-fase 3.2 — Resiliencia del DAG e Idempotencia

> **Rama:** `feat/3.2-airflow-resilience`
> **Fecha:** 6 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 3.2

---

## Estado General del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **0.1** | Inicialización y Dependencias | ✅ Completada |
| **0.2** | Contenedorización Base | ✅ Completada |
| **1.1** | Core de Dominio y Puertos Hexagonales | ✅ Completada |
| **1.2** | Implementación de Patrones GoF | ✅ Completada |
| **2.1** | Adaptadores de BigQuery y Storage | ✅ Completada |
| **2.2** | Entrypoint y Observabilidad Estructurada | ✅ Completada |
| **3.1** | DAG Base y Deferrable Operators | ✅ Completada |
| **3.2** | Resiliencia del DAG e Idempotencia | ✅ Completada |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Robustecer la tubería de orquestación en Apache Airflow (Cloud Composer) para dotarla de tolerancia a fallos transitorios e idempotencia. Esto se logra mediante el control de flujo condicional antes del entrenamiento pesado (evitando consumo innecesario de GPUs), la clasificación estructurada de excepciones de infraestructura (nativas de Airflow) y la integración de alertas automatizadas de Slack para monitorizar el estado de salud del pipeline.

---

## Entregables Creados y Modificados

### 1. `dags/solar_training_pipeline.py` — Resiliencia y Notificaciones Integradas [MODIFICADO]

Se extendió el DAG base de la Fase 3.1 incorporando la lógica de resiliencia requerida por el PRD (§4):
- **Tarea Previa `check_retraining_trigger`:** Nueva tarea del tipo `PythonOperator` que evalúa las condiciones de reentrenamiento basadas en el error medio (MAE) del modelo estable actual y la existencia de datos.
- **Clasificación de Excepciones:**
  - `AirflowException`: Se eleva ante fallos de conexión o timeouts de red transitorios consultando BigQuery, forzando reintentos con exponencial backoff configurados en `default_args`.
  - `AirflowFailException`: Se eleva ante fallos estructurales irreversibles (ej. columnas faltantes en el esquema de origen), interrumpiendo el flujo inmediatamente sin gastar reintentos inútiles.
  - `AirflowSkipException`: Se eleva si el modelo estable actual mantiene un MAE óptimo (< 0.05) o los datos no ameritan entrenamiento, omitiendo la tarea pesada downstream en Vertex AI.
- **Alertas de Slack Resilientes:** Se implementó el callback `notify_slack` utilizando `requests` y la conexión nativa `BaseHook` de Airflow para evitar acoplamientos con proveedores externos (`apache-airflow-providers-slack`). El callback está protegido contra fallos para evitar que la caída de Slack afecte al pipeline principal.
- **Exponential Backoff:** Configurado en `default_args` con reintentos progresivos hasta un máximo de 30 minutos de retraso.

### 2. `tests/dags/test_solar_training_pipeline.py` — Cobertura de Resiliencia [MODIFICADO]

Se añadieron pruebas unitarias para cubrir todas las nuevas ramificaciones del flujo de control:
- **Estructura y Dependencias:** Verifica que el DAG contenga exactamente dos tareas y que `check_retraining_trigger` sea el upstream inmediato de `train_transformer_bilstm`.
- **Comportamiento ante Excepciones:** Mocks de variables de control que afirman el correcto lanzamiento de `AirflowSkipException`, `AirflowFailException`, y `AirflowException` bajo las condiciones descritas de red y estructura de datos.
- **Integridad de Slack Callbacks:** Pruebas unitarias con mocking de `requests.post` y `BaseHook.get_connection` que aseguran que el formateador genera el payload correcto y es tolerante a errores (silent warning) si Slack no está accesible.

---

## Resultados del Test Suite (146 Tests en Verde)

Se ejecutó la suite de tests unitarios y de integración de todo el proyecto obteniendo un éxito rotundo:

```
tests/dags/test_solar_training_pipeline.py::test_dag_imports_without_errors PASSED [  8%]
tests/dags/test_solar_training_pipeline.py::test_dag_exists PASSED       [ 16%]
tests/dags/test_solar_training_pipeline.py::test_operator_configurations PASSED [ 25%]
tests/dags/test_solar_training_pipeline.py::test_check_retraining_trigger_force_skip PASSED [ 33%]
tests/dags/test_solar_training_pipeline.py::test_check_retraining_trigger_force_fail PASSED [ 41%]
tests/dags/test_solar_training_pipeline.py::test_check_retraining_trigger_force_error PASSED [ 50%]
tests/dags/test_solar_training_pipeline.py::test_check_retraining_trigger_network_error PASSED [ 58%]
tests/dags/test_solar_training_pipeline.py::test_check_retraining_trigger_structural_error PASSED [ 66%]
tests/dags/test_solar_training_pipeline.py::test_check_retraining_trigger_low_mae_skips PASSED [ 75%]
tests/dags/test_solar_training_pipeline.py::test_check_retraining_trigger_high_mae_runs PASSED [ 83%]
tests/dags/test_solar_training_pipeline.py::test_notify_slack_success PASSED [ 91%]
tests/dags/test_solar_training_pipeline.py::test_notify_slack_tolerant_to_exceptions PASSED [100%]

======================== 146 passed, 24 warnings in 3.02s ========================
```

---

## ⏭️ Próximo Paso: Fase 4 — Despliegue de Endpoints y Monitorización

**Rama:** `feat/4.1-model-registry-aliasing`

Lo que se creará:
- Evaluación Champion vs Challenger integrada en el DAG de Airflow.
- Orquestación de Version Aliasing en el Model Registry de Vertex AI para promover automáticamente modelos superiores a `stable`.
