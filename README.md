# ☀️ Serverless Solar MLOps

**Sistema de Pronóstico Fotovoltaico Automatizado con MLOps en Google Cloud Platform**

---

## Abstract

Este proyecto implementa un sistema empresarial de **MLOps de estado del arte** para la operacionalización de un modelo **Transformer Bi-LSTM** orientado al pronóstico de generación de energía solar fotovoltaica.

El pipeline automatiza el ciclo completo de vida del modelo de machine learning:

1. **Ingesta orientada a eventos** — Los datos estructurados de la Capa Oro en BigQuery disparan automáticamente el reentrenamiento vía Cloud Pub/Sub.
2. **Entrenamiento asíncrono** — Cloud Composer (Apache Airflow) orquesta trabajos de entrenamiento en Vertex AI mediante Deferrable Operators, evitando el bloqueo de recursos.
3. **Evaluación Champion vs Challenger** — El modelo candidato se compara científicamente (MAE/RMSE) contra el modelo en producción.
4. **Despliegue condicional** — Si el candidato demuestra superioridad, se promueve automáticamente mediante Version Aliasing en Vertex AI Model Registry.

### Principios de Diseño

| Principio | Implementación |
|-----------|---------------|
| **Arquitectura Hexagonal** | Separación estricta entre dominio matemático (`domain/`) e infraestructura GCP (`adapters/`) |
| **Patrones GoF** | Factory (instanciación dinámica), Strategy (funciones de pérdida), Adapter (conexiones externas) |
| **Observabilidad** | Logging estructurado JSON obligatorio, métricas basadas en registros en Cloud Monitoring |
| **Tolerancia a fallos** | Bloques `try-except` obligatorios con Exponential Backoff y clasificación de excepciones Airflow |
| **Reproducibilidad** | Dependencias ancladas, contenedores inmutables multi-stage, entorno virtual aislado |

### Stack Tecnológico

```
PyTorch · Pandas · BigQuery · Vertex AI · Cloud Composer (Airflow)
Cloud Pub/Sub · Cloud Monitoring · Docker · Pydantic · structlog
```

---

## 📐 Arquitectura del Proyecto

```
serverless-solar-mlops/
├── .env.example                    # Plantilla de variables de entorno
├── .gitignore                      # Protección de datos y secretos
├── Dockerfile.training             # Imagen multi-stage para Vertex AI
├── docker-compose.yml              # Emulación local de Vertex AI
├── requirements-train.txt          # Dependencias ancladas (pip)
│
├── src/                            # ─── Código fuente principal ───
│   ├── domain/                     # Core matemático (sin deps de GCP)
│   │   ├── models/
│   │   │   ├── config.py           # ModelConfig, TrainingConfig (Pydantic)
│   │   │   └── transformer_bilstm.py  # Modelo Transformer Bi-LSTM (PyTorch)
│   │   ├── ports/
│   │   │   └── ports.py            # Interfaces abstractas (Hexagonal)
│   │   └── strategies/
│   │       └── evaluation.py       # Strategy Pattern: MAE, RMSE
│   │
│   ├── adapters/                   # Infraestructura / Conexiones GCP
│   │   ├── adapter_config.py       # Configuraciones Pydantic (BQ, Storage)
│   │   ├── data_adapters.py        # BigQueryTimeSeriesAdapter
│   │   ├── model_adapters.py       # VertexModelRepositoryAdapter, Checkpoints
│   │   ├── logging_adapter.py      # CloudStructuredLogFormatter + Divergencia
│   │   └── monitoring.py           # Log-based Metrics y Alert Policies
│   │
│   └── entrypoints/                # Puntos de entrada
│       ├── train.py                # ⭐ Entrypoint principal (Vertex AI)
│       └── factories.py            # Factory Pattern: Model, Loss
│
├── dags/                           # ─── Orquestación Airflow ───
│   └── solar_training_pipeline.py  # DAG completo del pipeline
│
├── tests/                          # ─── Tests unitarios ───
│   ├── domain/                     # Tests del dominio
│   ├── adapters/                   # Tests de adaptadores
│   └── dags/                       # Tests del DAG
│
├── local_volumes/                  # Volúmenes para emulación local
│   ├── model-output/               # Simula AIP_MODEL_DIR
│   ├── checkpoints/                # Simula AIP_CHECKPOINT_DIR
│   └── data/                       # Datos locales de prueba
│
└── docs/
    ├── prd/prd.md                  # Product Requirements Document
    └── advances/                   # Documentación por sub-fase
```

---

## 🚀 ¿Cómo funciona el sistema?

El sistema tiene **dos modos de ejecución**: local (desarrollo) y cloud (producción).

### Flujo completo en producción (GCP)

```
┌─────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│  BigQuery    │─────▶│  Cloud Pub/Sub        │─────▶│  Cloud Composer  │
│  Capa Oro    │      │  "gold-layer-updated" │      │  (Airflow DAG)   │
└─────────────┘      └──────────────────────┘      └────────┬─────────┘
                                                            │
                                                   ┌────────▼─────────┐
                                                   │ 1. check_trigger │
                                                   │ ¿Reentrenar?     │
                                                   └────────┬─────────┘
                                                            │ Sí
                                                   ┌────────▼─────────┐
                                                   │ 2. Vertex AI     │
                                                   │ Custom Training  │
                                                   │ (train.py)       │
                                                   └────────┬─────────┘
                                                            │
                                                   ┌────────▼─────────┐
                                                   │ 3. evaluate_model│
                                                   │ Champion vs      │
                                                   │ Challenger       │
                                                   └────────┬─────────┘
                                                            │ Challenger gana
                                                   ┌────────▼─────────┐
                                                   │ 4. Deploy a      │
                                                   │ Vertex Endpoint  │
                                                   │ (alias: stable)  │
                                                   └──────────────────┘
```

1. **Trigger**: Cuando la Capa Oro de BigQuery recibe datos nuevos, un mensaje a Cloud Pub/Sub dispara el DAG.
2. **Check**: Airflow evalúa si el modelo actual se ha degradado (MAE >= 0.05). Si no, omite el entrenamiento (`AirflowSkipException`).
3. **Train**: Se lanza un Custom Container Training Job en Vertex AI con el contenedor Docker que ejecuta `train.py`.
4. **Evaluate**: Se compara el Challenger vs el Champion. Si gana, se promueve a `stable` y se despliega en el Endpoint.

---

## 🖥️ Guía de Ejecución Local (Desarrollo)

### Prerrequisitos

- Python 3.11+
- Docker y Docker Compose
- Cuenta de GCP con acceso a BigQuery y Vertex AI (para ejecución cloud)

### 1. Configuración inicial

```bash
# Clonar el repositorio
git clone https://github.com/n0mercy95/serverless-solar-mlops.git
cd serverless-solar-mlops

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements-train.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con los valores reales de tu proyecto GCP
```

### 2. Ejecutar el entrenamiento localmente

Hay **dos formas** de ejecutar el entrenamiento en tu máquina local:

#### Opción A: Ejecución directa con Python

```bash
# Activar el entorno virtual
source venv/bin/activate

# Configurar el PYTHONPATH para que Python encuentre los módulos
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Ejecutar el entrypoint de entrenamiento
python src/entrypoints/train.py
```

> **Nota:** Este método requiere que tengas configurado tu `.env` con credenciales de GCP válidas, ya que `train.py` se conecta a BigQuery para obtener los datos y a GCS/Vertex AI para guardar checkpoints y el modelo final.

#### Opción B: Ejecución con Docker (recomendado)

Docker Compose emula el entorno de Vertex AI localmente, montando volúmenes que simulan los directorios de GCS:

```bash
# Construir la imagen
docker compose build training

# Ejecutar el entrenamiento
docker compose up training
```

Los artefactos se guardarán en:
- `local_volumes/model-output/` → Modelo entrenado (`model.pt`, `metadata.json`)
- `local_volumes/checkpoints/` → Checkpoints por época
- `local_volumes/tensorboard-logs/` → Logs de TensorBoard

### 3. Ejecutar los tests

```bash
# Con pytest directamente (desde el venv)
source venv/bin/activate
cd src && python -m pytest ../tests/ -v --noconftest

# O con Docker Compose
docker compose run --rm test
```

### 4. Ejecutar el DAG de Airflow (Cloud Composer)

El DAG en `dags/solar_training_pipeline.py` está diseñado para ejecutarse en **Cloud Composer** (managed Airflow). Para probarlo localmente:

```bash
# Verificar la sintaxis del DAG
source venv/bin/activate
python -c "from dags.solar_training_pipeline import dag; print(f'DAG {dag.dag_id} cargado con {len(dag.tasks)} tareas')"
```

Para desplegarlo en Cloud Composer, sube el archivo DAG al bucket de tu ambiente:

```bash
# Subir el DAG a Cloud Composer
gcloud composer environments storage dags import \
  --environment=solar-mlops-composer \
  --location=us-central1 \
  --source=dags/solar_training_pipeline.py
```

---

## ☁️ Guía de Ejecución en Producción (GCP)

### 1. Construir y subir la imagen Docker

```bash
# Autenticarse en Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Construir la imagen
docker build -f Dockerfile.training -t us-central1-docker.pkg.dev/TU_PROYECTO/solar-mlops/training:latest .

# Subir a Artifact Registry
docker push us-central1-docker.pkg.dev/TU_PROYECTO/solar-mlops/training:latest
```

### 2. Configurar Cloud Composer

```bash
# Crear el ambiente de Cloud Composer (si no existe)
gcloud composer environments create solar-mlops-composer \
  --location=us-central1 \
  --image-version=composer-2.9.10-airflow-2.9.3

# Configurar las variables de entorno en Composer
gcloud composer environments update solar-mlops-composer \
  --location=us-central1 \
  --update-env-variables=GCP_PROJECT_ID=TU_PROYECTO,GCP_REGION=us-central1,...

# Subir el DAG
gcloud composer environments storage dags import \
  --environment=solar-mlops-composer \
  --location=us-central1 \
  --source=dags/solar_training_pipeline.py
```

### 3. Configurar el trigger de Pub/Sub

```bash
# Crear el tópico
gcloud pubsub topics create gold-layer-updated

# Crear la suscripción que dispara el DAG
# (esto se configura en Cloud Composer como un trigger del DAG)
```

### 4. Provisionar métricas y alertas de Cloud Monitoring

```bash
# Exportar la configuración de monitorización como JSON
source venv/bin/activate
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python -c "
from adapters.monitoring import export_monitoring_config
config = export_monitoring_config('TU_PROYECTO', output_path='monitoring_config.json')
print(f'Exportadas {len(config[\"log_based_metrics\"])} métricas y {len(config[\"alert_policies\"])} alertas')
"

# O provisionarlas directamente (requiere permisos en GCP)
python -c "
from adapters.monitoring import provision_log_based_metrics, provision_alert_policies
provision_log_based_metrics('TU_PROYECTO')
provision_alert_policies('TU_PROYECTO')
"
```

---

## ⚙️ Variables de Entorno

Todas las variables se configuran en el archivo `.env` (ver `.env.example` como plantilla):

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `GCP_PROJECT_ID` | ID del proyecto de GCP | `mi-proyecto-gcp` |
| `GCP_REGION` | Región de GCP | `us-central1` |
| `BQ_DATASET_GOLD` | Dataset de BigQuery (Capa Oro) | `gold_layer` |
| `BQ_TABLE_TIMESERIES` | Tabla con series de tiempo | `solar_timeseries` |
| `VERTEX_STAGING_BUCKET` | Bucket de staging para Vertex AI | `gs://mi-bucket` |
| `TRAIN_EPOCHS` | Número de épocas de entrenamiento | `100` |
| `TRAIN_BATCH_SIZE` | Tamaño del batch | `64` |
| `TRAIN_LEARNING_RATE` | Learning rate | `0.001` |
| `TRAIN_SEQUENCE_LENGTH` | Ventana de entrada (horas) | `168` (7 días) |
| `TRAIN_FORECAST_HORIZON` | Horizonte de predicción (horas) | `24` |
| `ALERT_TRAIN_LOSS_THRESHOLD` | Umbral de divergencia de train_loss | `1.0` |
| `ALERT_VAL_LOSS_THRESHOLD` | Umbral de calidad mínima de val_loss | `0.5` |

---

## 📋 Estado del Proyecto

| Fase | Sub-fase | Descripción | Rama | Estado |
|------|----------|-------------|------|--------|
| **0** | **0.1** | Inicialización y Dependencias | `chore/0.1-project-setup` | ✅ Completada |
| **0** | **0.2** | Contenedorización Base | `chore/0.2-docker-setup` | ✅ Completada |
| **1** | **1.1** | Core de Dominio y Puertos Hexagonales | `feat/1.1-domain-core` | ✅ Completada |
| **1** | **1.2** | Implementación de Patrones GoF | `feat/1.2-gof-patterns` | ✅ Completada |
| **2** | **2.1** | Adaptadores de BigQuery y Storage | `feat/2.1-io-adapters` | ✅ Completada |
| **2** | **2.2** | Entrypoint y Observabilidad Estructurada | `feat/2.2-training-logging` | ✅ Completada |
| **3** | **3.1** | DAG Base y Deferrable Operators | `feat/3.1-airflow-dag` | ✅ Completada |
| **3** | **3.2** | Resiliencia del DAG e Idempotencia | `feat/3.2-airflow-resilience` | ✅ Completada |
| **4** | **4.1** | Modelo Champion/Challenger y Aliasing | `feat/4.1-model-registry-aliasing` | ✅ Completada |
| **4** | **4.2** | Log-based Metrics y Alertas Automáticas | `feat/4.2-monitoring-alerts` | ✅ Completada |

---

## 📖 Documentación

- [PRD Completo](docs/prd/prd.md) — Product Requirements Document con arquitectura y plan de ejecución

### Avances por sub-fase

| Sub-fase | Documento |
|----------|-----------|
| 0.1 — Inicialización | [advance-0.1.md](docs/advances/advance-0.1.md) |
| 0.2 — Contenedorización | [advance-0.2.md](docs/advances/advance-0.2.md) |
| 1.1 — Dominio y Puertos | [advance-1.1.md](docs/advances/advance-1.1.md) |
| 1.2 — Patrones GoF | [advance-1.2.md](docs/advances/advance-1.2.md) |
| 2.1 — Adaptadores I/O | [advance-2.1.md](docs/advances/advance-2.1.md) |
| 2.2 — Entrypoint y Logging | [advance-2.2.md](docs/advances/advance-2.2.md) |
| 3.1 — DAG y Deferrable Ops | [advance-3.1.md](docs/advances/advance-3.1.md) |
| 3.2 — Resiliencia del DAG | [advance-3.2.md](docs/advances/advance-3.2.md) |
| 4.1 — Champion/Challenger | [advance-4.1.md](docs/advances/advance-4.1.md) |
| 4.2 — Monitoring y Alertas | [advance-4.2.md](docs/advances/advance-4.2.md) |

---

## 🧪 Ejecutar los tests

```bash
# Activar el entorno virtual
source venv/bin/activate

# Tests de dominio (sin dependencias de GCP)
cd src && python -m pytest ../tests/domain/ -v --noconftest

# Tests de adaptadores
cd src && python -m pytest ../tests/adapters/ -v --noconftest

# Tests del DAG de Airflow
cd src && python -m pytest ../tests/dags/ -v --noconftest

# Todos los tests con cobertura
cd src && python -m pytest ../tests/ -v --noconftest --cov=. --cov-report=term-missing
```

---

> *Proyecto en desarrollo activo — Última actualización: 7 de Julio, 2026*
