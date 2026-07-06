# ============================================================
# solar_training_pipeline.py — Airflow DAG de Entrenamiento Continuo
# Serverless Solar MLOps | Sub-fase 3.1
# ============================================================
# DAG de Apache Airflow que orquesta el entrenamiento de nuestro
# modelo Transformer Bi-LSTM en Vertex AI de forma asíncrona (diferible).
# ============================================================

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomContainerTrainingJobOperator,
)

# --- Recuperación de Parámetros de Configuración ---
# Usamos variables de entorno con fallbacks razonables del PRD
# para evitar consultas a la base de datos de Airflow en tiempo de parseo.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "tu-proyecto-gcp")
REGION = os.environ.get("GCP_REGION", "us-central1")
STAGING_BUCKET = os.environ.get("VERTEX_STAGING_BUCKET", "gs://tu-bucket-staging")
SERVICE_ACCOUNT = os.environ.get("VERTEX_SERVICE_ACCOUNT", "")

# Imagen de Docker construida (Sub-fase 0.2)
CONTAINER_URI = os.environ.get(
    "VERTEX_TRAINING_CONTAINER_URI",
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/solar-mlops/training:latest"
)

# BigQuery Capa Oro
BQ_DATASET_GOLD = os.environ.get("BQ_DATASET_GOLD", "gold_layer")
BQ_TABLE_TIMESERIES = os.environ.get("BQ_TABLE_TIMESERIES", "solar_timeseries")

# Hiperparámetros de entrenamiento
TRAIN_EPOCHS = os.environ.get("TRAIN_EPOCHS", "100")
TRAIN_BATCH_SIZE = os.environ.get("TRAIN_BATCH_SIZE", "64")
TRAIN_LEARNING_RATE = os.environ.get("TRAIN_LEARNING_RATE", "0.001")
TRAIN_SEQUENCE_LENGTH = os.environ.get("TRAIN_SEQUENCE_LENGTH", "168")
TRAIN_FORECAST_HORIZON = os.environ.get("TRAIN_FORECAST_HORIZON", "24")

# Configuración del Modelo
MODEL_TYPE = os.environ.get("MODEL_TYPE", "transformer_bilstm")
MODEL_D_MODEL = os.environ.get("MODEL_D_MODEL", "128")
MODEL_N_HEADS = os.environ.get("MODEL_N_HEADS", "8")
MODEL_N_ENCODER_LAYERS = os.environ.get("MODEL_N_ENCODER_LAYERS", "3")
MODEL_LSTM_HIDDEN_SIZE = os.environ.get("MODEL_LSTM_HIDDEN_SIZE", "256")
MODEL_DROPOUT = os.environ.get("MODEL_DROPOUT", "0.1")

# Instancia de Tensorboard en GCP
VERTEX_TENSORBOARD_INSTANCE = os.environ.get("VERTEX_TENSORBOARD_INSTANCE", "")

# Argumentos base del DAG
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

# --- Definición del DAG ---
with DAG(
    dag_id="solar_training_pipeline",
    default_args=default_args,
    description="Pipeline de entrenamiento continuo para el modelo de predicción solar",
    schedule=None,  # Event-driven
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["mlops", "solar", "vertex_ai"],
) as dag:

    # Mapeo de variables de entorno para inyectar en el contenedor de Vertex AI
    env_variables = {
        "ENVIRONMENT": "production",
        "LOG_LEVEL": "INFO",
        "STRUCTURED_LOGGING": "true",
        "GCP_PROJECT_ID": PROJECT_ID,
        "BQ_DATASET_GOLD": BQ_DATASET_GOLD,
        "BQ_TABLE_TIMESERIES": BQ_TABLE_TIMESERIES,
        "AIP_MODEL_DIR": f"{STAGING_BUCKET}/model-output",
        "AIP_CHECKPOINT_DIR": f"{STAGING_BUCKET}/checkpoints",
        "AIP_TENSORBOARD_LOG_DIR": f"{STAGING_BUCKET}/tensorboard-logs",
        "TRAIN_EPOCHS": str(TRAIN_EPOCHS),
        "TRAIN_BATCH_SIZE": str(TRAIN_BATCH_SIZE),
        "TRAIN_LEARNING_RATE": str(TRAIN_LEARNING_RATE),
        "TRAIN_SEQUENCE_LENGTH": str(TRAIN_SEQUENCE_LENGTH),
        "TRAIN_FORECAST_HORIZON": str(TRAIN_FORECAST_HORIZON),
        "MODEL_TYPE": str(MODEL_TYPE),
        "MODEL_D_MODEL": str(MODEL_D_MODEL),
        "MODEL_N_HEADS": str(MODEL_N_HEADS),
        "MODEL_N_ENCODER_LAYERS": str(MODEL_N_ENCODER_LAYERS),
        "MODEL_LSTM_HIDDEN_SIZE": str(MODEL_LSTM_HIDDEN_SIZE),
        "MODEL_DROPOUT": str(MODEL_DROPOUT),
    }

    # Creamos el trabajo de Custom Training en Vertex AI usando Deferrable Operator
    train_transformer_bilstm = CreateCustomContainerTrainingJobOperator(
        task_id="train_transformer_bilstm",
        display_name="solar-transformer-bilstm-training-{{ ts_nodash.lower() }}",
        container_uri=CONTAINER_URI,
        project_id=PROJECT_ID,
        region=REGION,  # Usar region en lugar de location
        staging_bucket=STAGING_BUCKET,
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type=None,  # Modificar si se requiere GPU (ej. "NVIDIA_TESLA_T4")
        accelerator_count=0,
        service_account=SERVICE_ACCOUNT or None,
        environment_variables=env_variables,
        tensorboard=VERTEX_TENSORBOARD_INSTANCE or None,
        deferrable=True,  # Habilita el comportamiento diferible asíncrono (Triggerer)
    )
