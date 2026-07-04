# 📊 Avance del Proyecto — Serverless Solar MLOps

> **Documento vivo** que registra el progreso fase a fase del sistema MLOps de pronóstico fotovoltaico.
> Referencia: [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md)

---

## Estado General del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **0.1** | Inicialización y Dependencias | ✅ Completada |
| **0.2** | Contenedorización Base | ⏳ Pendiente |
| **1.1** | Core de Dominio y Puertos Hexagonales | ⏳ Pendiente |
| **1.2** | Implementación de Patrones GoF | ⏳ Pendiente |
| **2.1** | Adaptadores de BigQuery y Storage | ⏳ Pendiente |
| **2.2** | Entrypoint y Observabilidad Estructurada | ⏳ Pendiente |
| **3.1** | DAG Base y Deferrable Operators | ⏳ Pendiente |
| **3.2** | Resiliencia del DAG e Idempotencia | ⏳ Pendiente |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## ✅ Sub-fase 0.1: Inicialización y Dependencias

**Rama sugerida:** `chore/0.1-project-setup`
**Fecha de completación:** 4 de Julio, 2026

### Objetivo

Establecer el entorno virtual, archivo `.env` base para secretos, configuración estricta de `.gitignore` para proteger datos, y la lista de dependencias ancladas (`requirements-train.txt`).

### Entregables Creados

#### 1. `.gitignore` — Protección de Datos y Secretos

Archivo de exclusión configurado con protección estricta en múltiples capas:

| Categoría | Qué se protege |
|-----------|---------------|
| **Secretos** | `.env`, `*.pem`, `*.key`, service account JSONs |
| **Datos/Modelos** | `*.csv`, `*.parquet`, `*.pt`, `*.pth`, `checkpoints/` |
| **Python** | `__pycache__/`, `*.pyc`, `venv/`, `dist/`, `build/` |
| **GCP Credentials** | `*-service-account.json`, `application_default_credentials.json` |
| **IDE** | `.vscode/`, `.idea/`, `.DS_Store` |
| **IaC** | `.terraform/`, `*.tfstate`, `*.tfvars` |

> [!IMPORTANT]
> Se incluye `!.env.example` para permitir versionar la plantilla de referencia sin exponer secretos reales.

#### 2. `.env.example` — Plantilla de Variables de Entorno

Archivo plantilla que documenta **todas** las variables de entorno necesarias para el proyecto, organizado en secciones:

- **GCP Base**: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_ZONE`
- **BigQuery (Capa Oro)**: `BQ_DATASET_GOLD`, `BQ_TABLE_TIMESERIES`
- **Vertex AI**: `VERTEX_STAGING_BUCKET`, `VERTEX_TENSORBOARD_INSTANCE`, `VERTEX_SERVICE_ACCOUNT`
- **Hiperparámetros de Entrenamiento**: Épocas, batch size, learning rate, sequence length, forecast horizon
- **Configuración del Modelo**: Tipo de modelo, dimensiones del transformer, heads, capas LSTM
- **Cloud Composer**: Entorno y ubicación
- **Observabilidad**: Nivel de log y logging estructurado
- **Pub/Sub**: Tópico y suscripción para triggers de reentrenamiento

**Uso:**
```bash
cp .env.example .env
# Editar .env con los valores reales del proyecto GCP
```

#### 3. `requirements-train.txt` — Dependencias Ancladas (Pinned)

Todas las versiones están **fijadas** (`==`) para garantizar reproducibilidad bit-a-bit entre desarrollo local y Vertex AI:

| Categoría | Paquetes | Versión |
|-----------|----------|---------|
| **Deep Learning** | `torch`, `torchvision` | 2.3.1, 0.18.1 |
| **Datos** | `pandas`, `numpy`, `pyarrow` | 2.2.2, 1.26.4, 16.1.0 |
| **GCP SDKs** | `bigquery`, `storage`, `aiplatform`, `logging`, `pubsub` | Últimas estables |
| **Validación** | `pydantic`, `pydantic-settings`, `python-dotenv` | 2.8.2, 2.4.0, 1.0.1 |
| **Logging** | `structlog` | 24.4.0 |
| **Testing** | `pytest`, `pytest-cov`, `pytest-mock` | 8.2.2, 5.0.0, 3.14.0 |
| **Utilidades** | `tqdm`, `scikit-learn` | 4.66.5, 1.5.1 |

> [!NOTE]
> Se eligió `structlog` como complemento para implementar el `CloudStructuredLogFormatter` requerido por el PRD en la Sub-fase 2.2.

#### 4. Entorno Virtual Python (`venv/`)

Creado con `python3 -m venv venv` en la raíz del proyecto. Excluido de Git por `.gitignore`.

#### 5. Estructura de Directorios `src/` (Scaffolding Hexagonal)

Se creó el árbol de directorios siguiendo la **Arquitectura Hexagonal** definida en el PRD:

```
src/
├── __init__.py
├── domain/                     # CORE MATEMÁTICO AISLADO
│   ├── __init__.py
│   ├── models/                 # Transformer Bi-LSTM (PyTorch)
│   │   └── __init__.py
│   ├── ports/                  # Interfaces abstractas
│   │   └── __init__.py
│   └── strategies/             # Strategy: MAE, RMSE
│       └── __init__.py
├── adapters/                   # INFRAESTRUCTURA (BigQuery, Vertex)
│   └── __init__.py
└── entrypoints/                # Punto de entrada Vertex AI
    └── __init__.py
```

> [!TIP]
> Cada `__init__.py` incluye un comentario descriptivo del propósito de la capa, facilitando la navegación para nuevos contribuidores.

### Estructura del Repositorio Actual

```
serverless-solar-mlops/
├── .gitignore                  ← [NUEVO] Protección de datos/secretos
├── .env.example                ← [NUEVO] Plantilla de variables
├── README.md                   ← (existente)
├── requirements-train.txt      ← [NUEVO] Dependencias ancladas
├── venv/                       ← [NUEVO] Entorno virtual (excluido de Git)
├── docs/
│   ├── prd/
│   │   └── prd.md              ← PRD del proyecto
│   └── advances/
│       └── advance-0.1.md      ← [NUEVO] Este documento
└── src/                        ← [NUEVO] Scaffolding hexagonal
    ├── domain/
    │   ├── models/
    │   ├── ports/
    │   └── strategies/
    ├── adapters/
    └── entrypoints/
```

---

## ⏭️ Próximo Paso: Sub-fase 0.2 — Contenedorización Base

**Rama:** `chore/0.2-docker-setup`

Lo que se creará:
- `Dockerfile.training` — Imagen multi-stage minimalista
- `docker-compose.yml` — Emulación local de volúmenes de Vertex AI
- Validación de que el contenedor compila y ejecuta correctamente

---

> *Última actualización: 4 de Julio, 2026*
