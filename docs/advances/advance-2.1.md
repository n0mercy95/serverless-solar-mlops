# 📊 Avance Sub-fase 2.1 — Adaptadores de BigQuery y Storage

> **Rama:** `feat/2.1-io-adapters`
> **Fecha:** 6 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 2.1

---

## Estado General del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **0.1** | Inicialización y Dependencias | ✅ Completada |
| **0.2** | Contenedorización Base | ✅ Completada |
| **1.1** | Core de Dominio y Puertos Hexagonales | ✅ Completada |
| **1.2** | Implementación de Patrones GoF | ✅ Completada |
| **2.1** | Adaptadores de BigQuery y Storage | ✅ Completada |
| **2.2** | Entrypoint y Observabilidad Estructurada | ⏳ Pendiente |
| **3.1** | DAG Base y Deferrable Operators | ⏳ Pendiente |
| **3.2** | Resiliencia del DAG e Idempotencia | ⏳ Pendiente |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Implementar los **Adaptadores estructurales concretos** (patrón GoF Adapter) que cumplen los contratos del dominio definidos en la Sub-fase 1.1, integrando el SDK de Google Cloud para extraer series de tiempo de BigQuery y persistir artefactos de modelo/checkpoints en el filesystem (compatible con Vertex AI).

---

## Entregables Creados

### 1. `src/adapters/adapter_config.py` — Configuración Pydantic

Esquemas de validación estricta para la configuración de los adaptadores:

- **`BigQueryConfig`**: Encapsula `project_id`, `dataset`, `table`, `feature_columns`, `target_column`, `timestamp_column`, y `split_ratio`. Lee valores por defecto desde variables de entorno (`GCP_PROJECT_ID`, `BQ_DATASET_GOLD`, `BQ_TABLE_TIMESERIES`). Incluye validadores para prevenir project_id vacío y para garantizar que el target esté incluido en las features.

- **`StorageConfig`**: Encapsula `model_dir` y `checkpoint_dir`, leyendo de `AIP_MODEL_DIR` y `AIP_CHECKPOINT_DIR` (variables inyectadas por Vertex AI en producción).

---

### 2. `src/adapters/data_adapters.py` — BigQueryTimeSeriesAdapter

Implementación concreta de `DataPort` con el siguiente pipeline interno:

```
BigQuery SQL Query → DataFrame → numpy array → Sliding Windows → Train/Val Split → PyTorch Tensors
```

#### Características clave:
- **Query SQL parametrizada**: Extrae todas las features de la Capa Oro ordenadas cronológicamente
- **Ventanas deslizantes (Sliding Windows)**: Transforma la serie temporal cruda en pares `(X, y)` donde `X` = ventana de entrada `(seq_len, n_features)` e `y` = horizonte de predicción `(forecast_horizon,)` usando solo la columna target
- **Cache interna**: La query se ejecuta una sola vez; llamadas subsecuentes a `load_training_data()` o `load_validation_data()` usan el cache
- **Inyección de dependencia**: El `bigquery.Client` es inyectable para testing con mocks
- **Try-except obligatorio** (PRD §4) con logging estructurado JSON en todas las operaciones de I/O

---

### 3. `src/adapters/model_adapters.py` — Adaptadores de Modelo y Checkpoints

Implementa dos puertos de la arquitectura hexagonal:

#### `VertexModelRepositoryAdapter` → `ModelRepositoryPort`

```
{model_dir}/
├── model.pt          # state_dict serializado con torch.save()
└── metadata.json     # métricas, config, timestamp (JSON)
```

- **`save_model(model, metadata)`**: Serializa `state_dict` + metadatos JSON. Incluye función recursiva `_make_json_serializable()` que convierte tensores PyTorch y arrays numpy a tipos nativos Python
- **`load_model(model_path)`**: Carga el `state_dict` usando `weights_only=True` (seguridad contra pickle attacks)
- **`load_metadata(model_path)`**: Lee el JSON de metadatos

#### `VertexCheckpointAdapter` → `CheckpointPort`

- **`save_checkpoint(state, epoch)`**: Persiste el estado completo como `checkpoint_epoch_NNN.pt`
- **`load_latest_checkpoint()`**: Escanea el directorio, filtra por regex `checkpoint_epoch_(\d+)\.pt$`, y carga el de epoch más alto. Retorna `None` si no hay checkpoints (primer entrenamiento)

**Principio de diseño**: Ambos adaptadores usan el filesystem como capa de abstracción. En producción, Vertex AI monta GCS como filesystem local, por lo que el código funciona **idénticamente** en local y en la nube sin código GCS-specific.

---

### 4. Tests Unitarios y Cobertura — 127 Tests Totales, 96% Cobertura Global

Se agregaron **67 nuevos tests** organizados en 3 archivos:

```
tests/adapters/
├── __init__.py                          ← [NUEVO] Scaffolding
├── test_adapter_config.py               ← [NUEVO] 22 tests de validación Pydantic
├── test_data_adapters.py                ← [NUEVO] 21 tests de BigQuery + windowing
└── test_model_adapters.py               ← [NUEVO] 24 tests de modelo + checkpoints
```

#### Detalles de las Nuevas Pruebas

| Archivo | Tests | Cobertura | Técnicas |
|---------|-------|-----------|----------|
| `test_adapter_config.py` | 22 | 100% | Validación Pydantic, `monkeypatch` para env vars |
| `test_data_adapters.py` | 21 | 98% | `unittest.mock` para BigQuery, numpy assertions |
| `test_model_adapters.py` | 24 | 87% | `tmp_path` de pytest, `torch.equal()` para roundtrip |

#### Highlights:
- **Roundtrip de pesos**: Verifica que `save_model → load_model` produce tensores idénticos bit a bit con `torch.equal()`
- **Windowing determinista**: Valida que `X[0]` contiene exactamente `data[0:seq_len]` y `y[0]` contiene `data[seq_len:seq_len+horizon, target_idx]`
- **Cache de datos**: Confirma que múltiples llamadas a `load_training_data()` ejecutan la query de BigQuery solo una vez
- **Ordenamiento de checkpoints**: Guarda checkpoints en orden aleatorio `[1, 3, 2, 5, 4]` y verifica que `load_latest_checkpoint()` retorna epoch 5
- **Serialización JSON**: Verifica conversión recursiva de tensores PyTorch y arrays numpy a tipos nativos

---

## Estructura del Repositorio Actualizada

```
serverless-solar-mlops/
├── docs/
│   └── advances/
│       ├── advance-0.1.md
│       ├── advance-0.2.md
│       ├── advance-1.1.md
│       ├── advance-1.2.md
│       └── advance-2.1.md               ← [NUEVO] Este documento
├── tests/
│   ├── adapters/
│   │   ├── __init__.py                  ← [NUEVO] Scaffolding
│   │   ├── test_adapter_config.py       ← [NUEVO] Tests config Pydantic
│   │   ├── test_data_adapters.py        ← [NUEVO] Tests BigQuery adapter
│   │   └── test_model_adapters.py       ← [NUEVO] Tests model/checkpoint
│   ├── domain/
│   │   ├── test_config.py
│   │   ├── test_ports.py
│   │   ├── test_strategies.py
│   │   └── test_transformer_bilstm.py
│   └── entrypoints/
│       └── test_factories.py
└── src/
    ├── adapters/
    │   ├── __init__.py                  ← [ACTUALIZADO] Re-exports completos
    │   ├── adapter_config.py            ← [NUEVO] Esquemas Pydantic
    │   ├── data_adapters.py             ← [NUEVO] BigQueryTimeSeriesAdapter
    │   └── model_adapters.py            ← [NUEVO] Model + Checkpoint adapters
    ├── domain/                          (sin cambios — OCP respetado)
    └── entrypoints/                     (sin cambios)
```

---

## Resultados de Tests

```
================================ tests coverage ================================
Name                                       Stmts   Miss  Cover   Missing
------------------------------------------------------------------------
src/adapters/__init__.py                       4      0   100%
src/adapters/adapter_config.py                35      0   100%
src/adapters/data_adapters.py                 85      2    98%
src/adapters/model_adapters.py               110     14    87%
src/domain/models/config.py                   19      0   100%
src/domain/models/transformer_bilstm.py       40      0   100%
src/domain/ports/ports.py                     24      0   100%
src/domain/strategies/loss_strategies.py      35      0   100%
src/entrypoints/factories.py                  25      0   100%
------------------------------------------------------------------------
TOTAL                                        386     16    96%
======================= 127 passed, 20 warnings in 1.28s =======================
```

---

## ⏭️ Próximo Paso: Sub-fase 2.2 — Entrypoint y Observabilidad Estructurada

**Rama:** `feat/2.2-training-logging`

Lo que se creará:
- `src/entrypoints/train.py` — Punto de entrada para Vertex AI Custom Training
- `CloudStructuredLogFormatter` — Logging JSON compatible con Cloud Logging (implementa `MetricsLoggerPort`)
- Bloques `try-except` obligatorios con severity levels y tracing
- Tests de integración del pipeline de entrenamiento

---

> *Última actualización: 6 de Julio, 2026*
