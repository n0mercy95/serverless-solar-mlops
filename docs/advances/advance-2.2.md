# 📊 Avance Sub-fase 2.2 — Entrypoint y Observabilidad Estructurada

> **Rama:** `feat/2.2-training-logging`
> **Fecha:** 6 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 2.2

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
| **3.1** | DAG Base y Deferrable Operators | ⏳ Pendiente |
| **3.2** | Resiliencia del DAG e Idempotencia | ⏳ Pendiente |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Ensamblar los puertos, adaptadores, y fábricas construidos en las fases anteriores dentro de un **entrypoint unificado** (`train.py`) para Vertex AI Custom Training. Además, proveer de **logging estructurado JSON** que satisfaga los requisitos de observabilidad y genere *Log-based Metrics* en Google Cloud.

---

## Entregables Creados

### 1. `src/adapters/logging_adapter.py` — CloudMetricsLoggerAdapter

Implementación concreta de `MetricsLoggerPort`.
- Emplea `CloudStructuredLogFormatter`, el cual convierte cada registro emitido por `logging` de Python en un payload JSON puramente compatible con GCP.
- Toda métrica transmitida al parámetro `extra={"jsonPayload": ...}` es agregada al primer nivel del objeto JSON final, facilitando métricas automáticas.
- Implementa `log_epoch_metrics` y `log_training_complete`.

### 2. `src/entrypoints/train.py` — El Orquestador

El corazón del contenedor de Vertex AI, encargado de ejecutar el loop de Continuous Training.

#### Componentes del Orquestador:
1. **Configuración Pydantic**: Instancia `BigQueryConfig`, `StorageConfig`, `ModelConfig` y `TrainingConfig` de forma segura.
2. **Inyección de Módulos**: Construye los adaptadores y carga los datos `(X, y)` convirtiéndolos a `torch.utils.data.DataLoader`.
3. **Fábricas de Dominio**: Crea el `TransformerBiLSTM` dinámicamente y la función de pérdida a través de `ModelFactory` y `LossFactory`.
4. **Bucle de Épocas + Checkpoints**:
   - Escanea el almacenamiento buscando si hubo interrupciones. Si hay un *checkpoint* reciente, el estado del modelo y optimizador son restaurados, ajustando `start_epoch` dinámicamente.
   - En cada época, registra métricas vía `CloudMetricsLoggerAdapter`.
   - Implementa **Gradient Clipping** en el optimizador Adam para evitar la explosión de gradientes típicos en redes recurrentes (LSTM) y Transformers.
5. **Resiliencia Extrema**: Todo envuelto en bloques `try...except Exception as e` obligatorios para garantizar la captura del error en log JSON y aborto intencional con `sys.exit(1)`.

---

### 3. Tests Unitarios Integrados (134 Tests Totales, 95% Cobertura Global)

Se agregaron pruebas para consolidar la confiabilidad del orquestador:

- **`test_logging_adapter.py`**:
  - Verifica que el log emitido es serializable a JSON.
  - Comprueba la propagación correcta del objeto `jsonPayload` extra y sus severidades (INFO, ERROR, etc.).

- **`test_train.py`**:
  - Testea el **Happy Path** del entrenamiento, comprobando la iteración de época por época y las llamadas al `CloudMetricsLoggerAdapter`.
  - Simula una reanudación **Resumes From Checkpoint**, verificando que el inicio del entrenamiento salta las épocas que ya han concluido.
  - Inyecta un error crítico simulado en la inicialización (vía `mock.side_effect`), afirmando que se genera un `sys.exit(1)` con captura JSON sin dañar el proceso de recolección de logs.

---

## Arquitectura Hexagonal y Tolerancia a Fallos Finalizada

En este punto, el repositorio cumple a cabalidad las normas del PRD:
1. **Tolerancia a fallos**: Guardado de checkpoints en cada época de Vertex AI y captura global de excepciones.
2. **Abstracción OCP**: `train.py` nunca llama a librerías de GCP; delega a `BigQueryTimeSeriesAdapter` o `VertexModelRepositoryAdapter`.
3. **Orientado a Objetos**: Instanciación dinámica. Si mañana necesitamos reemplazar la función `MAE` por `RMSE`, solo cambiamos la cadena de texto de entrada a la factoría.

---

## ⏭️ Próximo Paso: Fase 3 — Orquestación en Cloud Composer

**Sub-fase 3.1: DAG Base y Deferrable Operators**

Con el código empaquetable y testado finalizado, ahora pasaremos al orquestador Apache Airflow (Cloud Composer) para:
- Escribir un `DAG` que despache de manera asincrónica este código a Vertex AI.
- Utilizar `CreateCustomContainerTrainingJobOperator(deferrable=True)`.

---

> *Última actualización: 6 de Julio, 2026*
