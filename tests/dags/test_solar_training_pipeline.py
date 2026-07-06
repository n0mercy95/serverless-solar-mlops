# ============================================================
# test_solar_training_pipeline.py — Tests del DAG de Airflow
# Serverless Solar MLOps | Sub-fase 3.2
# ============================================================
# Valida la estructura del DAG, integridad sintáctica, dependencias,
# comportamiento ante excepciones de Airflow y envío de notificaciones.
# ============================================================

import os
from unittest.mock import MagicMock, patch
import pytest
from airflow.models import DagBag
from airflow.exceptions import AirflowException, AirflowFailException, AirflowSkipException
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomContainerTrainingJobOperator,
)


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    """Fixture para cargar el DagBag de Airflow en la carpeta dags/."""
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    dag_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dags")
    db = DagBag(dag_folder=dag_dir, include_examples=False)
    return db


def test_dag_imports_without_errors(dagbag: DagBag) -> None:
    """Valida que el DAG se cargue sin errores sintácticos o de importación."""
    assert not dagbag.import_errors, f"Errores al importar el DAG: {dagbag.import_errors}"


def test_dag_exists(dagbag: DagBag) -> None:
    """Verifica que el DAG con ID 'solar_training_pipeline' exista y tenga la estructura correcta."""
    dag_id = "solar_training_pipeline"
    assert dag_id in dagbag.dags
    
    dag = dagbag.dags[dag_id]
    assert len(dag.tasks) == 2
    
    # Comprobar tareas esperadas
    task_ids = [t.task_id for t in dag.tasks]
    assert "check_retraining_trigger" in task_ids
    assert "train_transformer_bilstm" in task_ids
    
    # Comprobar dependencias: check_retraining_trigger >> train_transformer_bilstm
    check_task = dag.get_task("check_retraining_trigger")
    train_task = dag.get_task("train_transformer_bilstm")
    
    assert train_task in check_task.downstream_list
    assert check_task in train_task.upstream_list


def test_operator_configurations(dagbag: DagBag) -> None:
    """Comprueba que el operador de Vertex AI esté correctamente instanciado y sea diferible."""
    dag = dagbag.dags["solar_training_pipeline"]
    task = dag.get_task("train_transformer_bilstm")
    
    assert isinstance(task, CreateCustomContainerTrainingJobOperator)
    assert task.deferrable is True
    assert task.replica_count == 1
    assert task.machine_type == "n1-standard-4"
    assert task.region == "us-central1"
    
    # Verificar variables de entorno inyectadas
    env_vars = task.environment_variables
    assert env_vars is not None
    assert env_vars["ENVIRONMENT"] == "production"
    assert env_vars["STRUCTURED_LOGGING"] == "true"
    assert env_vars["MODEL_TYPE"] == "transformer_bilstm"


# --- Tests de la Lógica de Decisión del Trigger (check_retraining_trigger) ---

def test_check_retraining_trigger_force_skip() -> None:
    """Verifica que si la variable de fuerza está en 'skip', lance AirflowSkipException."""
    from solar_training_pipeline import check_retraining_trigger
    
    with patch.dict(os.environ, {"FORCE_RETRAINING_TRIGGER": "skip"}):
        with pytest.raises(AirflowSkipException, match="El modelo estable actual cumple con las métricas"):
            check_retraining_trigger()


def test_check_retraining_trigger_force_fail() -> None:
    """Verifica que si la variable de fuerza está en 'fail', lance AirflowFailException."""
    from solar_training_pipeline import check_retraining_trigger
    
    with patch.dict(os.environ, {"FORCE_RETRAINING_TRIGGER": "fail"}):
        with pytest.raises(AirflowFailException, match="Error estructural crítico"):
            check_retraining_trigger()


def test_check_retraining_trigger_force_error() -> None:
    """Verifica que si la variable de fuerza está en 'error', lance AirflowException."""
    from solar_training_pipeline import check_retraining_trigger
    
    with patch.dict(os.environ, {"FORCE_RETRAINING_TRIGGER": "error"}):
        with pytest.raises(AirflowException, match="Fallo de conexión transitorio simulado"):
            check_retraining_trigger()


def test_check_retraining_trigger_network_error() -> None:
    """Verifica que un error de red lance AirflowException (provoca reintento)."""
    from solar_training_pipeline import check_retraining_trigger
    
    with patch.dict(os.environ, {"SIMULATE_NETWORK_ERROR": "true"}):
        with pytest.raises(AirflowException, match="Falla de red transitoria"):
            check_retraining_trigger()


def test_check_retraining_trigger_structural_error() -> None:
    """Verifica que un error estructural lance AirflowFailException (falla inmediatamente)."""
    from solar_training_pipeline import check_retraining_trigger
    
    with patch.dict(os.environ, {"SIMULATE_STRUCTURAL_ERROR": "true"}):
        with pytest.raises(AirflowFailException, match="Error estructural o de configuración irrecoverable"):
            check_retraining_trigger()


def test_check_retraining_trigger_low_mae_skips() -> None:
    """Verifica que si el MAE del modelo actual es bajo (< 0.05), lance AirflowSkipException."""
    from solar_training_pipeline import check_retraining_trigger
    
    with patch.dict(os.environ, {"SIMULATE_CURRENT_MAE": "0.03"}):
        with pytest.raises(AirflowSkipException, match="La degradación del modelo aún no justifica"):
            check_retraining_trigger()


def test_check_retraining_trigger_high_mae_runs() -> None:
    """Verifica que si el MAE del modelo actual es alto (>= 0.05), pase la tarea sin excepciones."""
    from solar_training_pipeline import check_retraining_trigger
    
    with patch.dict(os.environ, {"SIMULATE_CURRENT_MAE": "0.06"}):
        # No debe lanzar ninguna excepción
        check_retraining_trigger()


# --- Tests de Notificaciones de Slack ---

@patch("solar_training_pipeline.BaseHook")
@patch("solar_training_pipeline.requests")
def test_notify_slack_success(mock_requests, mock_base_hook) -> None:
    """Verifica que notify_slack intente enviar la notificación con los parámetros del contexto."""
    from solar_training_pipeline import notify_slack
    
    # Configurar mock de Connection de Airflow
    mock_conn = MagicMock()
    mock_conn.host = "http://localhost/mock-slack-webhook"
    mock_base_hook.get_connection.return_value = mock_conn
    
    # Configurar mock de respuesta de HTTP requests
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.post.return_value = mock_response
    
    # Crear un contexto mockeado de Airflow
    mock_ti = MagicMock()
    mock_ti.task_id = "test_task"
    mock_ti.dag_id = "test_dag"
    
    context = {
        "task_instance": mock_ti,
        "execution_date": "2026-07-06T15:00:00"
    }
    
    notify_slack(context, "SUCCESS")
    
    mock_base_hook.get_connection.assert_called_with("slack_conn")
    mock_requests.post.assert_called_once()
    args, kwargs = mock_requests.post.call_args
    assert kwargs["json"]["text"] is not None
    assert "SUCCESS" in kwargs["json"]["text"]
    assert "test_task" in kwargs["json"]["text"]
    assert "test_dag" in kwargs["json"]["text"]


@patch("solar_training_pipeline.BaseHook")
@patch("solar_training_pipeline.requests")
def test_notify_slack_tolerant_to_exceptions(mock_requests, mock_base_hook) -> None:
    """Verifica que si requests explota, notify_slack capture la excepción y no la propague."""
    from solar_training_pipeline import notify_slack
    
    # Configurar mock de Connection de Airflow
    mock_conn = MagicMock()
    mock_conn.host = "http://localhost/mock-slack-webhook"
    mock_base_hook.get_connection.return_value = mock_conn
    
    # Simular caída de red en requests
    mock_requests.post.side_effect = Exception("Slack is down")
    
    mock_ti = MagicMock()
    mock_ti.task_id = "test_task"
    mock_ti.dag_id = "test_dag"
    
    context = {
        "task_instance": mock_ti,
        "execution_date": "2026-07-06T15:00:00"
    }
    
    # No debe levantar excepciones (tolerancia a fallos)
    notify_slack(context, "FAILED")
    
    mock_requests.post.assert_called_once()
