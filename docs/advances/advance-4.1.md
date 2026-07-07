# 📊 Avance Sub-fase 4.1 — Modelo Champion/Challenger y Aliasing

> **Rama:** `feat/4.1-model-registry-aliasing`
> **Fecha:** 6 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 4.1

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
| **4.1** | Modelo Champion/Challenger y Aliasing | ✅ Completada |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Implementar el paso analítico en Apache Airflow (Cloud Composer) que compara las métricas de precisión técnica (MAE representado por `val_loss`) entre el modelo Challenger recién entrenado y la versión del modelo actualmente en producción (Champion etiquetado como `stable` en el Vertex AI Model Registry). Si el Challenger es mejor (o si no existe un modelo estable previo), el sistema lo promueve asociándole el alias `stable`, lo despliega en el Endpoint de Vertex AI y retira de forma limpia las instancias anteriores para optimizar el cómputo y los costos.

---

## Entregables Creados y Modificados

### 1. `dags/solar_training_pipeline.py` — Evaluación Champion/Challenger y Despliegue en Endpoints [MODIFICADO]

Se extendió el pipeline de entrenamiento incorporando la tarea de evaluación de modelos:
*   **Función `evaluate_champion_challenger`:**
    *   **I/O Desacoplada y Mockeable:** Carga las métricas del Challenger desde `metadata.json` admitiendo rutas nativas en Cloud Storage (usando `GCSHook.download_as_byte_string`) así como el filesystem local (para simulación y pruebas locales).
    *   **Simulación de Entorno:** Permite forzar el resultado de la evaluación y los valores del Champion mediante variables de entorno (`FORCE_EVALUATION_RESULT`, `SIMULATE_CHAMPION_MAE`) permitiendo verificar toda la lógica sin depender de GCP.
    *   **Vertex AI Model Registry & Aliasing:** Sube el Challenger como una nueva versión en el Model Registry con el alias inicial `candidate`. Si supera al modelo `stable` actual, le asigna el alias `stable` (reubicando el tag automáticamente).
    *   **Endpoint Deployment & Cleanup:** Crea/recupera el Endpoint de Vertex AI, despliega el nuevo modelo estable con el 100% del tráfico asignado, y undeploya las versiones anteriores del endpoint de forma segura.
*   **Tarea `evaluate_model`:** Nueva tarea `PythonOperator` añadida downstream de `train_transformer_bilstm`.
*   **Flujo Secuencial:** Se actualizó la secuencia del pipeline a: `check_retraining_trigger >> train_transformer_bilstm >> evaluate_model`.

### 2. `tests/dags/test_solar_training_pipeline.py` — Pruebas de Evaluación y Aliasing [MODIFICADO]

Se incorporaron casos de prueba exhaustivos con mocking para asegurar la resiliencia y el comportamiento del nuevo operador:
*   **Integridad del Grafo:** Valida que el DAG contenga exactamente 3 tareas con el orden y dependencias correctas.
*   **Tests de Fuerza y Simulación:**
    *   `test_evaluate_champion_challenger_force_promote`: Verifica la promoción forzada cuando el Challenger es mejor.
    *   `test_evaluate_champion_challenger_force_reject`: Asegura que se eleve `AirflowSkipException` cuando el Challenger es peor, abortando el pipeline sin modificar el alias estable.
    *   `test_evaluate_champion_challenger_force_error`: Confirma el correcto lanzamiento de `AirflowException` ante fallos de conexión.
*   **Flujo Real con Mocks del SDK:**
    *   `test_evaluate_champion_challenger_real_sdk_flow`: Simula llamadas completas a las APIs de Vertex AI (`init`, `Model.list`, `Model.upload`, `ModelServiceClient.merge_version_aliases`, `Endpoint.list`, `Endpoint.deploy`, `Endpoint.undeploy`) afirmando que se ejecute la promoción analítica y el despliegue del modelo estable en el endpoint de forma exitosa.

---

## Resultados del Test Suite (150 Tests en Verde)

Se ejecutaron todas las pruebas de la base de código obteniendo cobertura total:

```
tests/adapters/test_adapter_config.py .....................              [ 14%]
tests/adapters/test_data_adapters.py ...................                 [ 26%]
tests/adapters/test_logging_adapter.py ....                              [ 29%]
tests/adapters/test_model_adapters.py ...........................        [ 47%]
tests/dags/test_solar_training_pipeline.py ................              [ 58%]
tests/domain/test_config.py ................                             [ 68%]
tests/domain/test_ports.py .........                                     [ 74%]
tests/domain/test_strategies.py ............                             [ 82%]
tests/domain/test_transformer_bilstm.py .............                    [ 91%]
tests/entrypoints/test_factories.py ..........                           [ 98%]
tests/entrypoints/test_train.py ...                                      [100%]

======================= 150 passed, 24 warnings in 2.73s =======================
```

---

## ⏭️ Próximo Paso: Sub-fase 4.2 — Log-based Metrics y Alertas Automáticas

**Rama:** `feat/4.2-monitoring-alerts`

En el siguiente paso se implementarán las métricas basadas en registros (Log-based Metrics) en Cloud Monitoring analizando el JSON estructurado emitido por el modelo, y configurando alertas tempranas para detectar divergencias o degradación algorítmica.
