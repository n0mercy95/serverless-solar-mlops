# ============================================================
# test_solar_training_pipeline.py — Tests del DAG de Airflow
# Serverless Solar MLOps | Sub-fase 3.1
# ============================================================
# Valida la estructura del DAG, integridad sintáctica, dependencias
# y parámetros del operador diferible para Vertex AI.
# ============================================================

import os
import pytest
from airflow.models import DagBag
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomContainerTrainingJobOperator,
)


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    """Fixture para cargar el DagBag de Airflow en la carpeta dags/."""
    # Desactivamos la carga de ejemplos de Airflow para acelerar el test
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    
    # Especificamos el path de dags
    dag_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dags")
    db = DagBag(dag_folder=dag_dir, include_examples=False)
    return db


def test_dag_imports_without_errors(dagbag: DagBag) -> None:
    """Valida que el DAG se cargue sin errores sintácticos o de importación."""
    assert not dagbag.import_errors, f"Errores al importar el DAG: {dagbag.import_errors}"


def test_dag_exists(dagbag: DagBag) -> None:
    """Verifica que el DAG con ID 'solar_training_pipeline' exista en el bag."""
    dag_id = "solar_training_pipeline"
    assert dag_id in dagbag.dags
    
    dag = dagbag.dags[dag_id]
    assert dag.dag_id == dag_id
    assert len(dag.tasks) == 1
    assert dag.tasks[0].task_id == "train_transformer_bilstm"


def test_operator_configurations(dagbag: DagBag) -> None:
    """Comprueba que el operador esté correctamente instanciado y sea diferible."""
    dag = dagbag.dags["solar_training_pipeline"]
    task = dag.get_task("train_transformer_bilstm")
    
    # Comprobar tipo de clase
    assert isinstance(task, CreateCustomContainerTrainingJobOperator)
    
    # Comprobar requerimiento crítico de la sub-fase: Deferrable=True
    assert task.deferrable is True
    
    # Verificar parámetros básicos del operador
    assert task.replica_count == 1
    assert task.machine_type == "n1-standard-4"
    assert task.accelerator_type is None
    assert task.accelerator_count == 0
    
    # Verificar que se estén inyectando las variables de entorno esperadas por train.py
    env_vars = task.environment_variables
    assert env_vars is not None
    assert "ENVIRONMENT" in env_vars
    assert env_vars["ENVIRONMENT"] == "production"
    assert "LOG_LEVEL" in env_vars
    assert "STRUCTURED_LOGGING" in env_vars
    assert env_vars["STRUCTURED_LOGGING"] == "true"
    
    # Variables de GCP y BigQuery
    assert "GCP_PROJECT_ID" in env_vars
    assert "BQ_DATASET_GOLD" in env_vars
    assert "BQ_TABLE_TIMESERIES" in env_vars
    
    # Variables de almacenamiento (checkpoints y salida)
    assert "AIP_MODEL_DIR" in env_vars
    assert "AIP_CHECKPOINT_DIR" in env_vars
    assert "AIP_TENSORBOARD_LOG_DIR" in env_vars
    
    # Hiperparámetros de modelo y entrenamiento
    assert "TRAIN_EPOCHS" in env_vars
    assert "TRAIN_BATCH_SIZE" in env_vars
    assert "TRAIN_LEARNING_RATE" in env_vars
    assert "TRAIN_SEQUENCE_LENGTH" in env_vars
    assert "TRAIN_FORECAST_HORIZON" in env_vars
    assert "MODEL_TYPE" in env_vars
    assert "MODEL_D_MODEL" in env_vars
    assert "MODEL_N_HEADS" in env_vars
    assert "MODEL_N_ENCODER_LAYERS" in env_vars
    assert "MODEL_LSTM_HIDDEN_SIZE" in env_vars
    assert "MODEL_DROPOUT" in env_vars
