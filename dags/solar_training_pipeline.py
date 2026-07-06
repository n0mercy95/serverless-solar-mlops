# ============================================================
# solar_training_pipeline.py — Airflow DAG de Entrenamiento Continuo
# Serverless Solar MLOps | Sub-fase 3.2 (Resiliencia e Idempotencia)
# ============================================================
# DAG de Apache Airflow que orquesta el entrenamiento de nuestro
# modelo Transformer Bi-LSTM en Vertex AI de forma asíncrona (diferible).
# Integra:
#   - Excepciones personalizadas para tolerancia a fallos transitorios
#     y estructurales.
#   - Tarea de validación previa al entrenamiento (check_retraining_trigger).
#   - Notificaciones a Slack en caso de éxito y fallo (sin dependencias externas).
# ============================================================

import os
import logging
import requests
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException, AirflowFailException, AirflowSkipException
from airflow.hooks.base import BaseHook
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomContainerTrainingJobOperator,
)

# --- Callback de Notificaciones de Slack ---
def notify_slack(context, status: str):
    """Envía una notificación a Slack sobre el estado de la tarea/DAG.
    
    Usa requests para no requerir dependencias de proveedores externos (como apache-airflow-providers-slack).
    Se obtiene el Webhook de la conexión 'slack_conn' o de la variable de entorno 'SLACK_WEBHOOK_URL'.
    Envuelto en try-except para tolerancia a fallos.
    """
    try:
        webhook_url = None
        
        # 1. Intentar leer desde la conexión 'slack_conn' de Airflow
        try:
            conn = BaseHook.get_connection("slack_conn")
            # El webhook puede estar en el campo host o en extra_dejson
            webhook_url = conn.host or (conn.extra_dejson.get("webhook_url") if conn.extra else None)
        except Exception:
            pass
            
        # 2. Fallback a variable de entorno
        if not webhook_url:
            webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
            
        if not webhook_url:
            logging.warning("No se pudo enviar notificación de Slack: no hay 'slack_conn' ni 'SLACK_WEBHOOK_URL'.")
            return
            
        task_id = context.get("task_instance").task_id
        dag_id = context.get("task_instance").dag_id
        execution_date = context.get("execution_date")
        
        emoji = "✅" if status == "SUCCESS" else "🚨"
        message = (
            f"{emoji} *Pipeline Status Alert*\n"
            f"*DAG:* `{dag_id}`\n"
            f"*Task:* `{task_id}`\n"
            f"*Execution Date:* `{execution_date}`\n"
            f"*Status:* `{status}`\n"
        )
        
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logging.info(f"Notificación de Slack ({status}) enviada con éxito para {task_id}.")
        else:
            logging.warning(f"Slack devolvió un código de estado {response.status_code}: {response.text}")
    except Exception as e:
        # Fallo tolerante y silencioso ante caídas de Slack
        logging.warning(f"No se pudo enviar la alerta de Slack: {e}. Continuando ejecución...")


# --- Tarea de Decisión de Reentrenamiento ---
def check_retraining_trigger(**context):
    """Valida si la degradación del modelo justifica un reentrenamiento en Vertex AI.
    
    Aplica las siguientes políticas de resiliencia:
    - AirflowException: Para fallos de red/conexión transitorios en BigQuery (provoca reintentos).
    - AirflowFailException: Para fallos estructurales irreversibles (falla el pipeline de inmediato).
    - AirflowSkipException: Si las reglas de negocio determinan que el reentrenamiento no es necesario.
    """
    # 1. Comprobar si hay disparadores forzados mediante variables de entorno (Testing)
    force_trigger = os.environ.get("FORCE_RETRAINING_TRIGGER", "").lower()
    if force_trigger == "skip":
        raise AirflowSkipException(
            "Regla de negocio: El modelo estable actual cumple con las métricas. Se omite el reentrenamiento."
        )
    elif force_trigger == "fail":
        raise AirflowFailException(
            "Error estructural crítico: Estructura del origen de datos modificada de forma incompatible."
        )
    elif force_trigger == "error":
        raise AirflowException(
            "Fallo de conexión transitorio simulado al consultar BigQuery."
        )

    # 2. Simulación de lógica de negocio y I/O
    try:
        # Simular comportamiento de fallos basado en variables de control
        if os.environ.get("SIMULATE_NETWORK_ERROR") == "true":
            raise ConnectionError("Timeout al conectar con BigQuery Capa Oro.")
            
        if os.environ.get("SIMULATE_STRUCTURAL_ERROR") == "true":
            raise ValueError("Columna 'power_output' no encontrada en el origen de datos.")

        # Lógica de decisión: evaluamos el MAE (Mean Absolute Error) del modelo actual
        # Si el error es menor a 0.05, el modelo actual es óptimo y se skip-ea.
        current_mae = float(os.environ.get("SIMULATE_CURRENT_MAE", "0.04"))
        
        if current_mae < 0.05:
            raise AirflowSkipException(
                f"El modelo estable en producción mantiene alta precisión (MAE: {current_mae:.4f} < 0.05). "
                "La degradación del modelo aún no justifica el consumo de GPUs para un reentrenamiento."
            )
            
        logging.info(
            f"Degradación de modelo confirmada (MAE: {current_mae:.4f} >= 0.05). "
            "Despachando trabajo de Vertex AI."
        )
        
    except (ConnectionError, TimeoutError) as conn_err:
        logging.error(f"Fallo transitorio en la consulta: {conn_err}")
        raise AirflowException(f"Falla de red transitoria: {conn_err}")
        
    except (ValueError, KeyError) as struct_err:
        logging.error(f"Fallo estructural crítico: {struct_err}")
        raise AirflowFailException(f"Error estructural o de configuración irrecoverable: {struct_err}")


# --- Parámetros de Configuración del DAG ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "tu-proyecto-gcp")
REGION = os.environ.get("GCP_REGION", "us-central1")
STAGING_BUCKET = os.environ.get("VERTEX_STAGING_BUCKET", "gs://tu-bucket-staging")
SERVICE_ACCOUNT = os.environ.get("VERTEX_SERVICE_ACCOUNT", "")

CONTAINER_URI = os.environ.get(
    "VERTEX_TRAINING_CONTAINER_URI",
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/solar-mlops/training:latest"
)

BQ_DATASET_GOLD = os.environ.get("BQ_DATASET_GOLD", "gold_layer")
BQ_TABLE_TIMESERIES = os.environ.get("BQ_TABLE_TIMESERIES", "solar_timeseries")

TRAIN_EPOCHS = os.environ.get("TRAIN_EPOCHS", "100")
TRAIN_BATCH_SIZE = os.environ.get("TRAIN_BATCH_SIZE", "64")
TRAIN_LEARNING_RATE = os.environ.get("TRAIN_LEARNING_RATE", "0.001")
TRAIN_SEQUENCE_LENGTH = os.environ.get("TRAIN_SEQUENCE_LENGTH", "168")
TRAIN_FORECAST_HORIZON = os.environ.get("TRAIN_FORECAST_HORIZON", "24")

MODEL_TYPE = os.environ.get("MODEL_TYPE", "transformer_bilstm")
MODEL_D_MODEL = os.environ.get("MODEL_D_MODEL", "128")
MODEL_N_HEADS = os.environ.get("MODEL_N_HEADS", "8")
MODEL_N_ENCODER_LAYERS = os.environ.get("MODEL_N_ENCODER_LAYERS", "3")
MODEL_LSTM_HIDDEN_SIZE = os.environ.get("MODEL_LSTM_HIDDEN_SIZE", "256")
MODEL_DROPOUT = os.environ.get("MODEL_DROPOUT", "0.1")

VERTEX_TENSORBOARD_INSTANCE = os.environ.get("VERTEX_TENSORBOARD_INSTANCE", "")

# Argumentos base del DAG con reintentos mediante Exponential Backoff y notificaciones automáticas
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "on_failure_callback": lambda ctx: notify_slack(ctx, "FAILED"),
}

# --- Definición del DAG ---
with DAG(
    dag_id="solar_training_pipeline",
    default_args=default_args,
    description="Pipeline de entrenamiento continuo y resiliente para el modelo de predicción solar",
    schedule=None,  # Event-driven
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["mlops", "solar", "vertex_ai", "resilience"],
) as dag:

    # 1. Tarea de decisión antes del entrenamiento
    check_trigger = PythonOperator(
        task_id="check_retraining_trigger",
        python_callable=check_retraining_trigger,
    )

    # Variables de entorno para inyectar en el contenedor de Vertex AI
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

    # 2. Tarea de entrenamiento en Vertex AI
    train_transformer_bilstm = CreateCustomContainerTrainingJobOperator(
        task_id="train_transformer_bilstm",
        display_name="solar-transformer-bilstm-training-{{ ts_nodash.lower() }}",
        container_uri=CONTAINER_URI,
        project_id=PROJECT_ID,
        region=REGION,
        staging_bucket=STAGING_BUCKET,
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type=None,
        accelerator_count=0,
        service_account=SERVICE_ACCOUNT or None,
        environment_variables=env_variables,
        tensorboard=VERTEX_TENSORBOARD_INSTANCE or None,
        deferrable=True,
        on_success_callback=lambda ctx: notify_slack(ctx, "SUCCESS"),
    )

    # Flujo de dependencias
    check_trigger >> train_transformer_bilstm
