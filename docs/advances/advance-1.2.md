# 📊 Avance Sub-fase 1.2 — Implementación de Patrones GoF

> **Rama:** `feat/1.2-gof-patterns`
> **Fecha:** 6 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 1.2

---

## Estado General del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **0.1** | Inicialización y Dependencias | ✅ Completada |
| **0.2** | Contenedorización Base | ✅ Completada |
| **1.1** | Core de Dominio y Puertos Hexagonales | ✅ Completada |
| **1.2** | Implementación de Patrones GoF | ✅ Completada |
| **2.1** | Adaptadores de BigQuery y Storage | ⏳ Pendiente |
| **2.2** | Entrypoint y Observabilidad Estructurada | ⏳ Pendiente |
| **3.1** | DAG Base y Deferrable Operators | ⏳ Pendiente |
| **3.2** | Resiliencia del DAG e Idempotencia | ⏳ Pendiente |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Implementar los patrones de diseño orientados a objetos **Strategy** (para funciones de pérdida y evaluación matemática) y **Factory** (para instanciación dinámica del modelo y estrategias), desacoplando los entrypoints de ejecución de los detalles internos de modelado y optimización.

---

## Entregables Creados

### 1. `src/domain/strategies/loss_strategies.py` — Patrón Strategy

Implementa la abstracción de funciones de pérdida mediante una interfaz abstracta común (`LossStrategy`), permitiendo alternar el criterio de optimización o cálculo de métricas de forma transparente:

```
          ┌──────────────────┐
          │  LossStrategy    │ (ABC)
          └────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│   MAE    │ │   MSE    │ │   RMSE   │ (Concrete Strategies)
└──────────┘ └──────────┘ └──────────┘
```

- **`MAELossStrategy`**: L1 Loss (`nn.L1Loss`). Robusta ante outliers.
- **`MSELossStrategy`**: L2 Loss (`nn.MSELoss`). Diferenciable y óptima para backpropagation.
- **`RMSELossStrategy`**: Root Mean Squared Error. Implementado de forma numéricamente estable sumando un término $\epsilon = 10^{-8}$ antes de la raíz para prevenir gradientes indefinidos (`NaN`) cuando el error llega a cero.

---

### 2. `src/entrypoints/factories.py` — Patrón Factory

Encapsula la creación dinámica de modelos y estrategias de pérdida basándose en strings de configuración.

#### `ModelFactory`
- Utiliza un patrón de registro dinámico (Registry).
- Permite añadir nuevos modelos en caliente usando `ModelFactory.register(name, model_class)`.
- Registra automáticamente el modelo base `TransformerBiLSTM` bajo el alias `"transformer_bilstm"`.
- Resiste diferencias de formato (ej. mayúsculas, minúsculas, espacios adicionales).

#### `LossFactory`
- Fábrica estática para instanciar las estrategias definidas (`mae`, `mse`, `rmse`).
- Valida los nombres de entrada y levanta excepciones claras (`ValueError`) si no se soporta el criterio seleccionado.

---

### 3. Tests Unitarios y Cobertura — 60 Tests Totales, 100% Cobertura

Se agregaron pruebas exhaustivas de correctitud matemática, propagación de gradientes, y comportamiento de las fábricas:

```
tests/
├── conftest.py
└── domain/
    ├── test_config.py
    ├── test_ports.py
    ├── test_strategies.py               ← [NUEVO] 10 tests unitarios de estrategias
    └── test_transformer_bilstm.py
tests/entrypoints/
    ├── __init__.py                      ← [NUEVO] Scaffolding de entrypoints tests
    └── test_factories.py                ← [NUEVO] 12 tests de fábricas dinámicas
```

#### Detalles de las Nuevas Pruebas
- **Correctitud Matemática**: Se validó el cálculo manual del MAE, MSE, y RMSE contra inputs de prueba vectorizados para asegurar precisión de cálculo de punto flotante.
- **Estabilidad de Gradientes**: Se demostró que `RMSELossStrategy` propaga gradientes válidos y previene `NaN` cuando el error es exactamente cero utilizando aserciones `torch.isnan()`.
- **Fábricas Flexibles**:
  - `ModelFactory` crea correctamente el `TransformerBiLSTM`.
  - Ignora mayúsculas, minúsculas y espacios.
  - Soporta el registro de clases de modelo personalizadas `ModelFactory.register()` en tiempo de ejecución.
  - `LossFactory` maneja correctamente la parametrización de sus estrategias conocidas y levanta excepciones controladas para estrategias inválidas.

---

## Estructura del Repositorio Actualizada

```
serverless-solar-mlops/
├── docs/
│   └── advances/
│       ├── advance-1.1.md
│       └── advance-1.2.md               ← [NUEVO] Este documento
├── tests/
│   ├── domain/
│   │   └── test_strategies.py           ← [NUEVO] Tests para estrategias
│   └── entrypoints/
│       ├── __init__.py
│       └── test_factories.py            ← [NUEVO] Tests para fábricas GoF
└── src/
    ├── domain/
    │   └── strategies/
    │       ├── __init__.py              ← [ACTUALIZADO] Re-exports de estrategias
    │       └── loss_strategies.py       ← [NUEVO] Implementaciones de Strategy
    └── entrypoints/
        ├── __init__.py                  ← [ACTUALIZADO] Re-exports de fábricas
        └── factories.py                 ← [NUEVO] Fábricas del modelo y pérdida
```

---

## ⏭️ Próximo Paso: Fase 2 — Configuración de Vertex AI Custom Training y Model Registry

**Sub-fase 2.1:** Adaptadores de BigQuery y Storage (`feat/2.1-io-adapters`)

Lo que se creará:
- Implementación de `BigQueryTimeSeriesAdapter` que implementa `DataPort`.
- Implementación de `GCSModelRepositoryAdapter` que implementa `ModelRepositoryPort`.
- Implementación de `GCSCheckpointAdapter` que implementa `CheckpointPort`.
- Pruebas de integración usando mocks del SDK de Google Cloud.

---

> *Última actualización: 6 de Julio, 2026*
