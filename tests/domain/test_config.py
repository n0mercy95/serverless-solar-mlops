# ============================================================
# test_config.py — Tests de Validación Pydantic
# Serverless Solar MLOps | Sub-fase 1.1
# ============================================================

import pytest
from pydantic import ValidationError

from domain.models.config import ModelConfig, TrainingConfig


class TestModelConfig:
    """Tests para ModelConfig — esquema de hiperparámetros del modelo."""

    def test_default_values(self) -> None:
        """Los valores por defecto deben coincidir con .env.example."""
        config = ModelConfig()
        assert config.n_features == 7
        assert config.d_model == 128
        assert config.n_heads == 8
        assert config.n_encoder_layers == 3
        assert config.lstm_hidden_size == 256
        assert config.dropout == 0.1
        assert config.sequence_length == 168
        assert config.forecast_horizon == 24

    def test_custom_valid_config(self) -> None:
        """Debe aceptar configuraciones personalizadas válidas."""
        config = ModelConfig(
            n_features=5,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            lstm_hidden_size=128,
            dropout=0.2,
            sequence_length=48,
            forecast_horizon=12,
        )
        assert config.d_model == 64
        assert config.n_heads == 4

    def test_d_model_not_divisible_by_n_heads_raises(self) -> None:
        """d_model no divisible por n_heads debe lanzar ValidationError."""
        with pytest.raises(ValidationError, match="divisible"):
            ModelConfig(d_model=100, n_heads=8)

    def test_d_model_divisible_by_n_heads_passes(self) -> None:
        """d_model divisible por n_heads debe ser válido."""
        config = ModelConfig(d_model=96, n_heads=8)
        assert config.d_model == 96

    def test_negative_n_features_raises(self) -> None:
        """n_features < 1 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(n_features=0)

    def test_negative_dropout_raises(self) -> None:
        """dropout < 0.0 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(dropout=-0.1)

    def test_dropout_one_raises(self) -> None:
        """dropout >= 1.0 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.0)

    def test_zero_dropout_is_valid(self) -> None:
        """dropout = 0.0 debe ser válido (sin regularización)."""
        config = ModelConfig(dropout=0.0)
        assert config.dropout == 0.0

    def test_negative_sequence_length_raises(self) -> None:
        """sequence_length < 1 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(sequence_length=0)

    def test_negative_forecast_horizon_raises(self) -> None:
        """forecast_horizon < 1 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(forecast_horizon=0)


class TestTrainingConfig:
    """Tests para TrainingConfig — esquema de hiperparámetros de entrenamiento."""

    def test_default_values(self) -> None:
        """Los valores por defecto deben coincidir con .env.example."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 0.001

    def test_custom_valid_config(self) -> None:
        """Debe aceptar configuraciones personalizadas válidas."""
        config = TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.0001,
        )
        assert config.epochs == 50

    def test_zero_epochs_raises(self) -> None:
        """epochs < 1 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)

    def test_zero_batch_size_raises(self) -> None:
        """batch_size < 1 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_zero_learning_rate_raises(self) -> None:
        """learning_rate <= 0 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)

    def test_negative_learning_rate_raises(self) -> None:
        """learning_rate < 0 debe lanzar ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.001)
