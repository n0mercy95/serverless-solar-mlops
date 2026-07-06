# ============================================================
# test_transformer_bilstm.py — Tests del Modelo
# Serverless Solar MLOps | Sub-fase 1.1
# ============================================================

import torch

from domain.models.config import ModelConfig
from domain.models.transformer_bilstm import (
    PositionalEncoding,
    TransformerBiLSTM,
)


class TestPositionalEncoding:
    """Tests para el encoding posicional sinusoidal."""

    def test_output_shape_matches_input(self) -> None:
        """La salida debe tener el mismo shape que la entrada."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
        x = torch.randn(2, 50, d_model)
        output = pe(x)
        assert output.shape == x.shape

    def test_encoding_adds_positional_info(self) -> None:
        """El encoding debe modificar la entrada (no ser identidad)."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, d_model)
        output = pe(x)
        # Con input de ceros, la salida debe ser no-cero (= el PE puro)
        assert not torch.allclose(output, x)

    def test_encoding_is_deterministic(self) -> None:
        """El encoding posicional debe ser determinístico (sin dropout)."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
        pe.eval()  # Desactivar dropout residual
        x = torch.randn(1, 10, d_model)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)

    def test_different_positions_get_different_encodings(self) -> None:
        """Posiciones distintas deben recibir encodings distintos."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, d_model)
        output = pe(x)
        # Posición 0 y posición 1 deben ser diferentes
        assert not torch.allclose(output[0, 0, :], output[0, 1, :])


class TestTransformerBiLSTM:
    """Tests para el modelo Transformer Bi-LSTM completo."""

    def test_output_shape(
        self, small_model_config: ModelConfig, sample_input_tensor: torch.Tensor
    ) -> None:
        """La salida debe ser (batch_size, forecast_horizon)."""
        model = TransformerBiLSTM(small_model_config)
        model.eval()
        with torch.no_grad():
            output = model(sample_input_tensor)
        assert output.shape == (4, small_model_config.forecast_horizon)

    def test_output_shape_with_default_config(self) -> None:
        """Verificar output shape con la configuración del PRD."""
        config = ModelConfig()
        model = TransformerBiLSTM(config)
        model.eval()
        x = torch.randn(2, config.sequence_length, config.n_features)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, config.forecast_horizon)

    def test_single_sample_batch(self, small_model_config: ModelConfig) -> None:
        """El modelo debe funcionar con batch_size=1."""
        model = TransformerBiLSTM(small_model_config)
        model.eval()
        x = torch.randn(1, small_model_config.sequence_length, small_model_config.n_features)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, small_model_config.forecast_horizon)

    def test_gradients_flow(
        self,
        small_model_config: ModelConfig,
        sample_input_tensor: torch.Tensor,
        sample_target_tensor: torch.Tensor,
    ) -> None:
        """Todos los parámetros deben recibir gradiente en backward."""
        model = TransformerBiLSTM(small_model_config)
        model.train()
        output = model(sample_input_tensor)
        loss = torch.nn.MSELoss()(output, sample_target_tensor)
        loss.backward()

        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Parámetro '{name}' no recibió gradiente"
                )
                if not torch.all(param.grad == 0):
                    params_with_grad += 1

        # La mayoría de parámetros debe recibir gradiente no-cero
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert params_with_grad > total_params * 0.8, (
            f"Solo {params_with_grad}/{total_params} parámetros "
            f"recibieron gradiente no-cero"
        )

    def test_count_parameters_is_positive(
        self, small_model_config: ModelConfig
    ) -> None:
        """count_parameters debe retornar un número positivo."""
        model = TransformerBiLSTM(small_model_config)
        n_params = model.count_parameters()
        assert n_params > 0

    def test_count_parameters_increases_with_larger_model(self) -> None:
        """Un modelo más grande debe tener más parámetros."""
        small_config = ModelConfig(
            n_features=3, d_model=16, n_heads=4,
            n_encoder_layers=1, lstm_hidden_size=32,
            sequence_length=24, forecast_horizon=6,
        )
        large_config = ModelConfig(
            n_features=3, d_model=64, n_heads=4,
            n_encoder_layers=3, lstm_hidden_size=128,
            sequence_length=24, forecast_horizon=6,
        )
        small_model = TransformerBiLSTM(small_config)
        large_model = TransformerBiLSTM(large_config)
        assert large_model.count_parameters() > small_model.count_parameters()

    def test_model_stores_config(self, small_model_config: ModelConfig) -> None:
        """El modelo debe almacenar su configuración para serialización."""
        model = TransformerBiLSTM(small_model_config)
        assert model.config == small_model_config

    def test_eval_mode_is_deterministic(
        self, small_model_config: ModelConfig
    ) -> None:
        """En modo eval, el modelo debe ser determinístico."""
        model = TransformerBiLSTM(small_model_config)
        model.eval()
        x = torch.randn(2, small_model_config.sequence_length, small_model_config.n_features)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_produce_different_outputs(
        self, small_model_config: ModelConfig
    ) -> None:
        """Entradas diferentes deben producir salidas diferentes."""
        model = TransformerBiLSTM(small_model_config)
        model.eval()
        x1 = torch.randn(2, small_model_config.sequence_length, small_model_config.n_features)
        x2 = torch.randn(2, small_model_config.sequence_length, small_model_config.n_features)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert not torch.allclose(out1, out2)
