# ============================================================
# test_factories.py — Tests para las Fábricas de Modelos y Pérdidas
# Serverless Solar MLOps | Sub-fase 1.2
# ============================================================

import pytest
import torch
import torch.nn as nn

from domain.models.config import ModelConfig
from domain.models.transformer_bilstm import TransformerBiLSTM
from domain.strategies.loss_strategies import (
    MAELossStrategy,
    MSELossStrategy,
    RMSELossStrategy,
)
from entrypoints.factories import LossFactory, ModelFactory


class TestModelFactory:
    """Tests para verificar el patrón de diseño ModelFactory."""

    def test_creates_registered_model(self) -> None:
        """Debe crear correctamente un TransformerBiLSTM registrado por defecto."""
        config = ModelConfig(
            n_features=3,
            d_model=16,
            n_heads=2,
            lstm_hidden_size=16,
            sequence_length=12,
            forecast_horizon=3,
        )
        model = ModelFactory.create("transformer_bilstm", config)
        assert isinstance(model, TransformerBiLSTM)
        assert model.config == config

    def test_case_insensitive_creation(self) -> None:
        """La creación debe ignorar mayúsculas/minúsculas y espacios."""
        config = ModelConfig(d_model=16, n_heads=2)
        model = ModelFactory.create("  Transformer_BiLSTM  ", config)
        assert isinstance(model, TransformerBiLSTM)

    def test_unregistered_model_raises_value_error(self) -> None:
        """Solicitar un modelo no registrado debe levantar ValueError."""
        config = ModelConfig()
        with pytest.raises(
            ValueError, match="no está registrado en la fábrica"
        ):
            ModelFactory.create("non_existent_lstm", config)

    def test_register_new_model_class(self) -> None:
        """Debe permitir registrar y crear un modelo dinámicamente."""

        class CustomModel(nn.Module):
            def __init__(self, config: ModelConfig):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(config.n_features, config.forecast_horizon)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x[:, -1, :])

        # Registrar
        ModelFactory.register("custom_linear", CustomModel)

        # Crear y probar
        config = ModelConfig(n_features=5, forecast_horizon=2)
        model = ModelFactory.create("custom_linear", config)

        assert isinstance(model, CustomModel)
        assert model.config == config

        # Limpiar el registro para evitar efectos colaterales en otros tests
        if "custom_linear" in ModelFactory._registry:
            del ModelFactory._registry["custom_linear"]


class TestLossFactory:
    """Tests para verificar la fábrica LossFactory."""

    @pytest.mark.parametrize(
        "loss_name, expected_class",
        [
            ("mae", MAELossStrategy),
            ("mse", MSELossStrategy),
            ("rmse", RMSELossStrategy),
            ("  MAE  ", MAELossStrategy),
            ("Rmse", RMSELossStrategy),
        ],
    )
    def test_creates_supported_strategies(
        self, loss_name: str, expected_class: type
    ) -> None:
        """Debe crear las estrategias soportadas (case-insensitive)."""
        strategy = LossFactory.create(loss_name)
        assert isinstance(strategy, expected_class)

    def test_unsupported_strategy_raises_value_error(self) -> None:
        """Solicitar una estrategia no soportada debe levantar ValueError."""
        with pytest.raises(ValueError, match="no está soportada"):
            LossFactory.create("huber_loss")
