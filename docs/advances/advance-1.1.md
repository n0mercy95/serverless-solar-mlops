# 📊 Avance Sub-fase 1.1 — Core de Dominio y Puertos Hexagonales

> **Rama:** `feat/1.1-domain-core`
> **Fecha:** 6 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 1.1

---

## Estado General del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **0.1** | Inicialización y Dependencias | ✅ Completada |
| **0.2** | Contenedorización Base | ✅ Completada |
| **1.1** | Core de Dominio y Puertos Hexagonales | ✅ Completada |
| **1.2** | Implementación de Patrones GoF | ⏳ Pendiente |
| **2.1** | Adaptadores de BigQuery y Storage | ⏳ Pendiente |
| **2.2** | Entrypoint y Observabilidad Estructurada | ⏳ Pendiente |
| **3.1** | DAG Base y Deferrable Operators | ⏳ Pendiente |
| **3.2** | Resiliencia del DAG e Idempotencia | ⏳ Pendiente |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Aislar puramente la lógica algorítmica del Transformer Bi-LSTM (en PyTorch) bajo el directorio `domain/`, creando los puertos abstractos de salida (interfaces) para garantizar alta testabilidad desacoplada de la nube.

---

## Entregables Creados

### 1. `src/domain/models/config.py` — Esquemas Pydantic de Configuración

Dos esquemas con validación estricta que encapsulan todos los hiperparámetros del modelo:

#### `ModelConfig` — Hiperparámetros Arquitectónicos

| Campo | Default | Validación |
|-------|---------|------------|
| `n_features` | 7 | `≥ 1` |
| `d_model` | 128 | `≥ 1`, divisible por `n_heads` |
| `n_heads` | 8 | `≥ 1` |
| `n_encoder_layers` | 3 | `≥ 1` |
| `lstm_hidden_size` | 256 | `≥ 1` |
| `dropout` | 0.1 | `[0.0, 1.0)` |
| `sequence_length` | 168 | `≥ 1` (7 días × 24h) |
| `forecast_horizon` | 24 | `≥ 1` (24h adelante) |

> [!IMPORTANT]
> El `model_validator` verifica que `d_model % n_heads == 0` — una restricción matemática del Multi-Head Attention donde cada cabeza opera sobre `d_model // n_heads` dimensiones.

#### `TrainingConfig` — Hiperparámetros de Entrenamiento

| Campo | Default | Validación |
|-------|---------|------------|
| `epochs` | 100 | `≥ 1` |
| `batch_size` | 64 | `≥ 1` |
| `learning_rate` | 0.001 | `> 0.0` |

---

### 2. `src/domain/models/transformer_bilstm.py` — Modelo Híbrido

Arquitectura del `TransformerBiLSTM(nn.Module)`:

```
Input (batch, seq_len=168, n_features=7)
    │
    ▼
InputProjection (Linear: 7 → 128)
    │
    ▼
PositionalEncoding (sinusoidal, Vaswani et al. 2017)
    │
    ▼
TransformerEncoder (3 capas, 8 heads, d_model=128, FFN=512)
  └─ Pre-LayerNorm (más estable para entrenamiento)
    │
    ▼
BiLSTM (input=128, hidden=256, bidireccional)
  └─ Toma último timestep como resumen de secuencia
    │
    ▼
OutputHead (Linear: 512 → 24)
    │
    ▼
Output (batch, forecast_horizon=24)
```

#### Decisiones de Diseño

| Decisión | Justificación |
|----------|---------------|
| **Pre-LayerNorm** (`norm_first=True`) | Más estable numéricamente para entrenamiento de Transformers profundos |
| **`batch_first=True`** | Consistencia con la convención PyTorch moderna y el Bi-LSTM |
| **Positional Encoding como buffer** | Se mueve con `.to(device)` pero no consume gradientes |
| **Último timestep del Bi-LSTM** | Resumen comprimido de toda la secuencia para proyectar al horizonte |
| **`PositionalEncoding` como clase separada** | Reutilizable, testeable independientemente |
| **`count_parameters()` método** | Facilita logging estructurado del tamaño del modelo |

#### Componentes Internos

- **`PositionalEncoding`**: Encoding sinusoidal clásico con dropout, registrado como buffer no-entrenable
- **`InputProjection`**: `nn.Linear(n_features, d_model)` — proyecta al espacio del Transformer
- **`TransformerEncoder`**: `nn.TransformerEncoder` con capas `nn.TransformerEncoderLayer`
- **`BiLSTM`**: `nn.LSTM(bidirectional=True)` — captura dependencias locales
- **`OutputHead`**: `nn.Linear(lstm_hidden*2, forecast_horizon)` — proyección final

---

### 3. `src/domain/ports/ports.py` — Puertos Hexagonales Abstractos

Cuatro interfaces `ABC` que definen los contratos entre el dominio y la infraestructura:

| Puerto | Responsabilidad | Métodos |
|--------|----------------|---------|
| **`DataPort`** | Adquisición de datos de series de tiempo | `load_training_data()`, `load_validation_data()` |
| **`ModelRepositoryPort`** | Persistencia de artefactos del modelo | `save_model()`, `load_model()` |
| **`CheckpointPort`** | Tolerancia a fallos durante entrenamiento | `save_checkpoint()`, `load_latest_checkpoint()` |
| **`MetricsLoggerPort`** | Logging de métricas de entrenamiento | `log_epoch_metrics()`, `log_training_complete()` |

> [!NOTE]
> Los tipos de retorno usan `torch.Tensor` porque PyTorch es una dependencia del dominio matemático, no de infraestructura cloud. **Cero imports de `google.cloud.*` en `src/domain/`** — verificado.

Estos puertos serán implementados como adaptadores concretos en la Sub-fase 2.1:
- `DataPort` → `BigQueryTimeSeriesAdapter`
- `ModelRepositoryPort` → `ArtifactRegistryAdapter`
- `CheckpointPort` → `GCSCheckpointAdapter`
- `MetricsLoggerPort` → `CloudStructuredLogAdapter`

---

### 4. Tests Unitarios — 38 Tests, 100% Cobertura

```
tests/
├── __init__.py
├── conftest.py                          # Fixtures: configs, tensores dummy
└── domain/
    ├── __init__.py
    ├── test_config.py                   # 16 tests — validación Pydantic
    ├── test_transformer_bilstm.py       # 13 tests — modelo completo
    └── test_ports.py                    # 9 tests — puertos abstractos
```

#### Resultados

```
38 passed, 0 failed, 100% coverage (0.23s)
```

| Suite | Tests | Qué verifica |
|-------|-------|-------------|
| `test_config.py` | 16 | Defaults del PRD, configs custom, `d_model % n_heads`, bounds negativos, edge cases (dropout=0, dropout=1) |
| `test_transformer_bilstm.py` | 13 | Output shapes, forward pass con config default y small, batch_size=1, flujo de gradientes, `count_parameters()`, determinismo en eval, inputs distintos → outputs distintos |
| `test_ports.py` | 9 | Instanciación directa de ABCs falla, implementaciones concretas mock funcionan, implementaciones parciales fallan |

---

## Estructura del Repositorio Actualizada

```
serverless-solar-mlops/
├── .dockerignore                        ← (fase 0.2)
├── .env.example                         ← (fase 0.1)
├── .gitignore                           ← (fase 0.2)
├── Dockerfile.training                  ← (fase 0.2)
├── README.md
├── docker-compose.yml                   ← (fase 0.2)
├── requirements-train.txt               ← (fase 0.1)
├── docs/
│   ├── advances/
│   │   ├── advance-0.1.md              ← (fase 0.1)
│   │   ├── advance-0.2.md              ← (fase 0.2)
│   │   └── advance-1.1.md              ← [NUEVO] Este documento
│   └── prd/
│       └── prd.md
├── local_volumes/                       ← (fase 0.2)
├── tests/                               ← [NUEVO] Suite de tests
│   ├── conftest.py
│   └── domain/
│       ├── test_config.py
│       ├── test_transformer_bilstm.py
│       └── test_ports.py
└── src/
    ├── domain/
    │   ├── models/
    │   │   ├── config.py                ← [NUEVO] Esquemas Pydantic
    │   │   └── transformer_bilstm.py    ← [NUEVO] Modelo Transformer Bi-LSTM
    │   ├── ports/
    │   │   └── ports.py                 ← [NUEVO] 4 puertos abstractos ABC
    │   └── strategies/                  ← (scaffolding, fase 1.2)
    ├── adapters/                        ← (scaffolding, fase 2.1)
    └── entrypoints/                     ← (scaffolding, fase 2.2)
```

---

## ⏭️ Próximo Paso: Sub-fase 1.2 — Implementación de Patrones GoF

**Rama:** `feat/1.2-gof-patterns`

Lo que se creará:
- `src/domain/strategies/loss_strategies.py` — Patrón Strategy: MAE, RMSE, MSE Loss
- `src/domain/strategies/evaluation_strategies.py` — Patrón Strategy: métricas de evaluación
- `src/entrypoints/factories.py` — Patrón Factory: instanciación dinámica del modelo
- Tests unitarios de strategies y factories

---

> *Última actualización: 6 de Julio, 2026*
