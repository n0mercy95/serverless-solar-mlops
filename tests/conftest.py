# ============================================================
# conftest.py — Fixtures Compartidas para Tests
# Serverless Solar MLOps | Sub-fase 1.1
# ============================================================

import pytest
import torch

from domain.models.config import ModelConfig, TrainingConfig


@pytest.fixture
def default_model_config() -> ModelConfig:
    """ModelConfig con valores por defecto del PRD."""
    return ModelConfig()


@pytest.fixture
def small_model_config() -> ModelConfig:
    """ModelConfig reducido para tests rápidos (menos parámetros)."""
    return ModelConfig(
        n_features=3,
        d_model=16,
        n_heads=4,
        n_encoder_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
        sequence_length=24,
        forecast_horizon=6,
    )


@pytest.fixture
def default_training_config() -> TrainingConfig:
    """TrainingConfig con valores por defecto del PRD."""
    return TrainingConfig()


@pytest.fixture
def sample_input_tensor(small_model_config: ModelConfig) -> torch.Tensor:
    """Tensor de entrada con dimensiones realistas para el modelo pequeño.

    Shape: (batch_size=4, sequence_length=24, n_features=3)
    """
    return torch.randn(
        4,
        small_model_config.sequence_length,
        small_model_config.n_features,
    )


@pytest.fixture
def sample_target_tensor(small_model_config: ModelConfig) -> torch.Tensor:
    """Tensor target con dimensiones del forecast horizon.

    Shape: (batch_size=4, forecast_horizon=6)
    """
    return torch.randn(4, small_model_config.forecast_horizon)
