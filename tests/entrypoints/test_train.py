# ============================================================
# test_train.py — Tests del Entrypoint
# Serverless Solar MLOps | Sub-fase 2.2
# ============================================================

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

# Usamos import dinámico dentro de los tests para no cargar env vars reales 
# antes de mockearlas, pero en pytest podemos inyectar las vars en un fixture.

@pytest.fixture(autouse=True)
def setup_env():
    """Configura las variables de entorno para que Pydantic no explote."""
    env_vars = {
        "GCP_PROJECT_ID": "test-project",
        "BQ_DATASET_GOLD": "test_dataset",
        "BQ_TABLE_TIMESERIES": "test_table",
        "AIP_MODEL_DIR": "/tmp/test_models",
        "AIP_CHECKPOINT_DIR": "/tmp/test_checkpoints",
        "MODEL_D_MODEL": "16",
        "MODEL_N_HEADS": "2",
        "TRAINING_EPOCHS": "2",
        "TRAINING_EPOCHS": "2",
        "TRAINING_BATCH_SIZE": "8"
    }
    with patch.dict(os.environ, env_vars):
        yield

@pytest.fixture
def mock_domain_configs():
    with patch("entrypoints.train.ModelConfig") as MockModel, \
         patch("entrypoints.train.TrainingConfig") as MockTraining:
        
        mock_model = MockModel.return_value
        mock_model.n_features = 8
        mock_model.d_model = 16
        mock_model.n_heads = 2
        mock_model.dropout = 0.1
        mock_model.n_encoder_layers = 1
        mock_model.lstm_hidden_size = 16
        mock_model.sequence_length = 168
        mock_model.forecast_horizon = 24
        mock_model.model_dump.return_value = {}
        
        mock_training = MockTraining.return_value
        mock_training.epochs = 2
        mock_training.batch_size = 8
        mock_training.learning_rate = 0.001
        mock_training.model_dump.return_value = {}
        
        yield (mock_model, mock_training)


@patch("entrypoints.train.BigQueryTimeSeriesAdapter")
@patch("entrypoints.train.VertexModelRepositoryAdapter")
@patch("entrypoints.train.VertexCheckpointAdapter")
@patch("entrypoints.train.logger")
def test_main_training_loop_success(
    mock_logger,
    mock_checkpoint,
    mock_model_repo,
    mock_bq_adapter,
    mock_domain_configs,
):
    """Prueba el ciclo de entrenamiento completo (happy path)."""
    
    # 1. Configurar los mocks de adaptadores
    # Mock data: shapes (n_samples, seq_len, features) -> (20, 24, 8)
    # y (n_samples, forecast) -> (20, 24)
    mock_X_train = torch.randn(20, 24, 8)
    mock_y_train = torch.randn(20, 24)
    mock_X_val = torch.randn(10, 24, 8)
    mock_y_val = torch.randn(10, 24)
    
    # El adaptador de BQ devuelve esta data mockeada
    adapter_instance = mock_bq_adapter.return_value
    adapter_instance.load_training_data.return_value = (mock_X_train, mock_y_train)
    adapter_instance.load_validation_data.return_value = (mock_X_val, mock_y_val)
    
    # El adaptador de checkpoint dice que NO hay checkpoints
    ckpt_instance = mock_checkpoint.return_value
    ckpt_instance.load_latest_checkpoint.return_value = None
    
    # 2. Ejecutar la función main()
    from entrypoints.train import main
    main()
    
    # 3. Asserts
    # Debería haber guardado checkpoint por cada epoch (2 en total)
    assert ckpt_instance.save_checkpoint.call_count == 2
    
    # Debería haber llamado al menos una vez para guardar el mejor modelo
    repo_instance = mock_model_repo.return_value
    assert repo_instance.save_model.call_count >= 1
    
    # Validar que el logger haya emitido la culminación
    mock_logger.log_training_complete.assert_called_once()
    assert mock_logger.log_epoch_metrics.call_count == 2


@patch("entrypoints.train.BigQueryTimeSeriesAdapter")
@patch("entrypoints.train.VertexModelRepositoryAdapter")
@patch("entrypoints.train.VertexCheckpointAdapter")
@patch("entrypoints.train.logger")
def test_main_resumes_from_checkpoint(
    mock_logger,
    mock_checkpoint,
    mock_model_repo,
    mock_bq_adapter,
    mock_domain_configs,
):
    """Valida que si hay un checkpoint, se resume y las epochs se ajustan."""
    
    mock_X_train = torch.randn(20, 24, 8)
    mock_y_train = torch.randn(20, 24)
    mock_X_val = torch.randn(10, 24, 8)
    mock_y_val = torch.randn(10, 24)
    
    adapter_instance = mock_bq_adapter.return_value
    adapter_instance.load_training_data.return_value = (mock_X_train, mock_y_train)
    adapter_instance.load_validation_data.return_value = (mock_X_val, mock_y_val)
    
    # Simular que estamos en la época 1 y vamos a hacer la época 2
    from domain.models.config import ModelConfig
    from domain.models.transformer_bilstm import TransformerBiLSTM
    
    dummy_model = TransformerBiLSTM(ModelConfig(
        n_features=8,
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        lstm_hidden_size=16,
        sequence_length=168,
        forecast_horizon=24
    ))
    import torch.optim as optim
    dummy_opt = optim.Adam(dummy_model.parameters())

    ckpt_instance = mock_checkpoint.return_value
    ckpt_instance.load_latest_checkpoint.return_value = {
        "epoch": 1,
        "model_state_dict": dummy_model.state_dict(),
        "optimizer_state_dict": dummy_opt.state_dict(),
        "best_val_loss": 0.5,
    }
    
    # Ejecutamos main
    from entrypoints.train import main
    main()
    
    # Como ya hizo la 1, y el test tiene 2 epochs, solo debería entrenar 1 vez
    assert ckpt_instance.save_checkpoint.call_count == 1
    assert mock_logger.log_epoch_metrics.call_count == 1


@patch("sys.exit")
@patch("entrypoints.train.BigQueryConfig")
@patch("entrypoints.train.logger")
def test_main_catches_and_logs_exception(
    mock_logger,
    mock_bq_config,
    mock_sys_exit
):
    """Valida que cualquier excepción sea capturada por el logger y no haga crash sucio."""
    
    # Simulamos que al instanciar config, explota
    mock_bq_config.side_effect = ValueError("Error de red BQ")
    mock_sys_exit.side_effect = SystemExit(1)
    
    from entrypoints.train import main
    with pytest.raises(SystemExit):
        main()
    
    # Debe haber llamado sys.exit(1)
    mock_sys_exit.assert_called_with(1)
    
    # El logger _logger.error debe haber sido llamado con la exc_info
    assert mock_logger._logger.error.call_count == 1
    args, kwargs = mock_logger._logger.error.call_args
    assert "Fallo crítico" in args[0]
    assert kwargs["exc_info"] is True
