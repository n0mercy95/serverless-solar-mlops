# 📊 Avance Sub-fase 3.1 — DAG Base y Deferrable Operators

> **Rama:** `feat/3.1-airflow-dag`
> **Fecha:** 6 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 3.1

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
| **3.2** | Resiliencia del DAG e Idempotencia | ⏳ Pendiente |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Ensamblar el pipeline de orquestación de entrenamiento en Apache Airflow (Cloud Composer), utilizando **operadores diferibles (Deferrable Operators)** para despachar trabajos a Vertex AI. Esto evita que los workers del orquestador se bloqueen esperando la finalización del entrenamiento, optimizando costos y recursos de infraestructura.

---

## Entregables Creados

### 1. `dags/solar_training_pipeline.py` — Airflow DAG de Entrenamiento Continuo

Se implementó el pipeline utilizando las mejores prácticas de desarrollo en Airflow:
- **Evitar Consultas en Parseo:** Se eliminaron las consultas a la base de datos en tiempo de parseo (`Variable.get` en módulo superior) reemplazándolas por variables de entorno y fallbacks. Esto optimiza drásticamente el rendimiento del programador (Scheduler) y evita bloqueos.
- **Operador Diferible:** Se empleó `CreateCustomContainerTrainingJobOperator(deferrable=True)` para delegar el sondeo (polling) del entrenamiento al componente asíncrono **Triggerer** de Airflow.
- **Configuración Dinámica:** Se inyectan las credenciales, staging buckets, y la configuración de hiperparámetros directamente en las variables de entorno de Vertex AI.
- **Uso de Parámetros Nuevos:** Se reemplazó el obsoleto `schedule_interval` por `schedule=None` y `location` por `region`.

### 2. `src/entrypoints/train.py` — Configuración Dinámica de Hiperparámetros

Se actualizó el entrypoint de entrenamiento para que los hiperparámetros inyectados desde el DAG sean aplicados dinámicamente en `ModelConfig` y `TrainingConfig`:
- Se lee desde `os.environ` variables como `MODEL_D_MODEL`, `MODEL_N_HEADS`, `TRAIN_EPOCHS`, etc.
- Se mantiene el desacoplamiento de la arquitectura hexagonal, ya que el dominio (`domain/`) no tiene dependencias directas con el sistema operativo ni variables de entorno.

### 3. `tests/dags/test_solar_training_pipeline.py` — Cobertura de Pruebas unitarias

Se implementó una suite de validación automatizada para el DAG:
- **`test_dag_imports_without_errors`**: Valida la integridad sintáctica y descarta fallos de importación del DAG.
- **`test_dag_exists`**: Valida que la estructura tenga el ID correspondiente (`solar_training_pipeline`) y contenga exactamente el task configurado.
- **`test_operator_configurations`**: Valida que se use la clase del operador correcto, que el parámetro `deferrable=True` esté activo, y comprueba la correspondencia exacta de variables de entorno inyectadas.

---

## Resultados del Test Suite (137 Tests en Verde)

Se ejecutó la suite completa de pruebas obteniendo un éxito rotundo:

```
tests/dags/test_solar_training_pipeline.py::test_dag_imports_without_errors PASSED [ 33%]
tests/dags/test_solar_training_pipeline.py::test_dag_exists PASSED       [ 66%]
tests/dags/test_solar_training_pipeline.py::test_operator_configurations PASSED [100%]

============================== 3 passed in 1.90s ===============================
```

Y la integración global con los 134 tests previos:
```
======================= 137 passed, 24 warnings in 2.92s =======================
```

---

## ⏭️ Próximo Paso: Sub-fase 3.2 — Resiliencia del DAG e Idempotencia

**Rama:** `feat/3.2-airflow-resilience`

Lo que se creará:
- Reglas de control de flujo con reintentos basados en `Exponential Backoff`.
- Gestión estructurada de excepciones utilizando excepciones nativas de Airflow (`AirflowException`, `AirflowFailException`, y `AirflowSkipException`) en el flujo.
- Integración de notificaciones del estado del pipeline.

---

> *Última actualización: 6 de Julio, 2026*
