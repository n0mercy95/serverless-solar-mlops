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


# --- Evaluación Champion vs Challenger y Aliasing ---
def evaluate_champion_challenger(**context):
    """
    Compara el modelo Challenger recién entrenado contra el Champion (stable) actual en Vertex AI.
    Si el Challenger es superior (MAE/val_loss más bajo) o no existe un stable anterior,
    lo promueve a 'stable' en el Vertex AI Model Registry y lo despliega en el Endpoint.
    """
    import os
    import json
    import logging
    from urllib.parse import urlparse
    from airflow.exceptions import AirflowException, AirflowFailException, AirflowSkipException

    staging_bucket = os.environ.get("VERTEX_STAGING_BUCKET", "gs://tu-bucket-staging")
    project_id = os.environ.get("GCP_PROJECT_ID", "tu-proyecto-gcp")
    region = os.environ.get("GCP_REGION", "us-central1")
    model_name = os.environ.get("VERTEX_MODEL_NAME", "solar_transformer_bilstm")
    endpoint_name = os.environ.get("VERTEX_ENDPOINT_NAME", "solar_transformer_bilstm_endpoint")
    serving_container = os.environ.get(
        "VERTEX_SERVING_CONTAINER_IMAGE_URI",
        "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest"
    )

    # 1. Leer las métricas del Challenger desde metadata.json
    metadata_uri = f"{staging_bucket}/model-output/metadata.json"
    challenger_val_loss = None
    
    try:
        if metadata_uri.startswith("gs://"):
            from airflow.providers.google.cloud.hooks.gcs import GCSHook
            hook = GCSHook()
            parsed = urlparse(metadata_uri)
            bucket_name = parsed.netloc
            object_name = parsed.path.lstrip('/')
            metadata_content = hook.download_as_byte_string(bucket_name, object_name)
            challenger_metadata = json.loads(metadata_content.decode('utf-8'))
        else:
            with open(metadata_uri, "r", encoding="utf-8") as f:
                challenger_metadata = json.load(f)
        
        challenger_val_loss = float(challenger_metadata.get("val_loss", 999.0))
        logging.info(f"Métricas del Challenger cargadas con éxito. MAE (val_loss): {challenger_val_loss:.6f}")
    except Exception as e:
        logging.error(f"Error cargando metadatos del Challenger: {e}")
        raise AirflowFailException(f"No se pudieron leer las métricas del Challenger: {e}")

    # 2. Lógica para tests / simulación
    force_result = os.environ.get("FORCE_EVALUATION_RESULT", "").lower()
    simulate_champion_mae = os.environ.get("SIMULATE_CHAMPION_MAE")
    
    if force_result == "error":
        raise AirflowException("Error de conexión simulado con Vertex AI.")
    
    if force_result in ["promote", "reject"] or simulate_champion_mae is not None:
        champion_val_loss = float(simulate_champion_mae) if simulate_champion_mae else 0.05
        logging.info(f"Simulando Champion con MAE: {champion_val_loss:.6f}")
        
        if force_result == "promote" or (force_result != "reject" and challenger_val_loss < champion_val_loss):
            logging.info("Resultado de simulación: Challenger supera al Champion. Promocionando a stable...")
            return "Promoted (Simulated)"
        else:
            logging.info("Resultado de simulación: Champion sigue siendo mejor. Omitiendo promoción.")
            raise AirflowSkipException("El modelo candidato no superó al modelo estable en producción.")

    # 3. Integración real con Vertex AI SDK
    try:
        from google.cloud import aiplatform
        from google.cloud import aiplatform_v1
        
        aiplatform.init(project=project_id, location=region)
        
        # Buscar si el modelo ya está en el registro
        model_resource_name = None
        models = aiplatform.Model.list(filter=f'display_name="{model_name}"', project=project_id, location=region)
        if models:
            model_resource_name = models[0].resource_name
            logging.info(f"Modelo existente encontrado en el registro: {model_resource_name}")
            
            # Obtener el Champion actual (stable)
            try:
                champion_model = aiplatform.Model(
                    model_name=f"{model_resource_name}@stable",
                    project=project_id,
                    location=region
                )
                if champion_model.version_description:
                    champion_metadata = json.loads(champion_model.version_description)
                    champion_val_loss = float(champion_metadata.get("val_loss", 999.0))
                else:
                    champion_val_loss = 999.0
                logging.info(f"Champion actual ('stable') cargado con éxito. MAE: {champion_val_loss:.6f}")
            except Exception as e:
                logging.warning(f"No se pudo obtener el Champion actual ('stable') o carece de metadatos: {e}. Asumiendo pérdida infinita.")
                champion_val_loss = float("inf")
        else:
            logging.info("No se encontró ningún modelo en el registro. Este será el Champion inicial.")
            champion_val_loss = float("inf")

        # Subir el Challenger como una nueva versión (con alias 'candidate')
        artifact_uri = f"{staging_bucket}/model-output"
        logging.info(f"Registrando Challenger en Vertex AI Model Registry desde {artifact_uri}...")
        
        version_description_json = json.dumps({"val_loss": challenger_val_loss})
        
        if model_resource_name:
            challenger_model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=artifact_uri,
                serving_container_image_uri=serving_container,
                parent_model=model_resource_name,
                version_aliases=["candidate"],
                version_description=version_description_json,
                project=project_id,
                location=region,
            )
        else:
            challenger_model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=artifact_uri,
                serving_container_image_uri=serving_container,
                version_aliases=["candidate"],
                version_description=version_description_json,
                project=project_id,
                location=region,
            )
            model_resource_name = challenger_model.resource_name
            
        logging.info(f"Challenger subido correctamente. Versión ID: {challenger_model.version_id}, Resource Name: {challenger_model.resource_name}")

        # Comparar y Promocionar
        if challenger_val_loss < champion_val_loss:
            logging.info(f"¡Challenger ({challenger_val_loss:.6f}) es mejor que Champion ({champion_val_loss:.6f})! Promocionando a 'stable'...")
            
            client = aiplatform_v1.ModelServiceClient(
                client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
            )
            
            target_version_resource = f"{model_resource_name}@{challenger_model.version_id}"
            logging.info(f"Asociando alias 'stable' a la versión: {target_version_resource}")
            client.merge_version_aliases(
                name=target_version_resource,
                version_aliases=["stable"]
            )
            
            # Desplegar al Endpoint de Vertex AI
            logging.info(f"Orquestando despliegue de la versión {challenger_model.version_id} al Endpoint...")
            
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"',
                project=project_id,
                location=region
            )
            if endpoints:
                endpoint = endpoints[0]
                logging.info(f"Endpoint existente encontrado: {endpoint.resource_name}")
            else:
                logging.info(f"Creando nuevo Endpoint: {endpoint_name}...")
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_name,
                    project=project_id,
                    location=region
                )
                logging.info(f"Endpoint creado: {endpoint.resource_name}")

            model_to_deploy = aiplatform.Model(f"{model_resource_name}@stable")
            
            logging.info("Iniciando deploy en el Endpoint...")
            deployed_model = endpoint.deploy(
                model=model_to_deploy,
                deployed_model_display_name=f"deployed-{model_name}",
                traffic_percentage=100,
                machine_type="n1-standard-2",
                min_replica_count=1,
                max_replica_count=1,
            )
            logging.info("Modelo desplegado con éxito en el Endpoint.")

            logging.info("Undeploying old models from endpoint to save resources...")
            for dm in endpoint.deployed_models:
                if dm.id != deployed_model.id:
                    logging.info(f"Undeploying model instance {dm.id} (model version {dm.model_id})...")
                    try:
                        endpoint.undeploy(deployed_model_id=dm.id)
                    except Exception as e:
                        logging.warning(f"No se pudo undeployar el modelo {dm.id}: {e}")
                        
            return "Promoted and Deployed"
        else:
            logging.info(f"Challenger ({challenger_val_loss:.6f}) no superó al Champion ({champion_val_loss:.6f}). No se actualiza 'stable'.")
            raise AirflowSkipException("El modelo candidato no superó al modelo estable en producción.")
            
    except AirflowSkipException:
        raise
    except Exception as e:
        logging.error(f"Fallo en la evaluación/promoción de Vertex AI: {e}", exc_info=True)
        raise AirflowException(f"Error en evaluación/promoción de Vertex AI: {e}")


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

    # 3. Tarea de evaluación Champion vs Challenger y despliegue/aliasing
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_champion_challenger,
    )

    # Flujo de dependencias
    check_trigger >> train_transformer_bilstm >> evaluate_model
