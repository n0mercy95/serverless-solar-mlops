# ============================================================
# transformer_bilstm.py — Modelo Transformer Bi-LSTM
# Serverless Solar MLOps | Sub-fase 1.1
# ============================================================
# Arquitectura híbrida para pronóstico de series de tiempo
# fotovoltaicas. El Transformer Encoder captura patrones
# globales mediante self-attention, y el Bi-LSTM refina las
# dependencias secuenciales bidireccionales.
#
# NOTA: Este módulo pertenece al CORE MATEMÁTICO AISLADO.
#       No contiene ninguna dependencia de Google Cloud.
# ============================================================

import math

import torch
import torch.nn as nn

from domain.models.config import ModelConfig


class PositionalEncoding(nn.Module):
    """Encoding posicional sinusoidal (Vaswani et al., 2017).

    Inyecta información de posición absoluta en los embeddings
    de entrada del Transformer, permitiendo que el mecanismo de
    atención distinga entre posiciones en la secuencia temporal.

    Fórmula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Dimensión del espacio de embedding.
        max_len: Longitud máxima de secuencia soportada.
        dropout: Tasa de dropout aplicada después del encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crear la matriz de positional encoding como buffer (no parámetro)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) para broadcasting con batch
        pe = pe.unsqueeze(0)

        # Registrar como buffer: se mueve con .to(device) pero no se entrena
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica positional encoding a la entrada.

        Args:
            x: Tensor de entrada con shape (batch_size, seq_len, d_model).

        Returns:
            Tensor con positional encoding sumado, mismo shape que la entrada.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBiLSTM(nn.Module):
    """Modelo híbrido Transformer Encoder + Bi-LSTM para pronóstico
    de series de tiempo fotovoltaicas.

    Arquitectura:
        1. InputProjection: Proyecta n_features → d_model
        2. PositionalEncoding: Encoding sinusoidal clásico
        3. TransformerEncoder: Self-attention sobre la secuencia temporal
        4. BiLSTM: Captura dependencias secuenciales bidireccionales
        5. OutputHead: Proyección final → forecast_horizon

    La separación Transformer + Bi-LSTM permite que el modelo:
    - Capture dependencias globales (patrones estacionales, ciclos solares)
      mediante el mecanismo de atención del Transformer.
    - Refine dependencias locales secuenciales (transiciones hora a hora)
      mediante las gates del LSTM bidireccional.

    Args:
        config: Instancia de ModelConfig con hiperparámetros validados.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Stage 1: Proyección de entrada ---
        # Transforma el espacio de features al espacio del Transformer
        self.input_projection = nn.Linear(config.n_features, config.d_model)

        # --- Stage 2: Positional Encoding ---
        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.sequence_length,
            dropout=config.dropout,
        )

        # --- Stage 3: Transformer Encoder ---
        # Cada capa tiene: Multi-Head Attention → Feed-Forward → LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,  # Input/Output: (batch, seq, feature)
            norm_first=True,  # Pre-LayerNorm (más estable para entrenamiento)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_encoder_layers,
        )

        # --- Stage 4: Bi-LSTM ---
        # Procesa la salida del Transformer bidireccionalmente
        self.bilstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # Una sola capa LSTM, dropout no aplica
        )

        # Dropout entre Bi-LSTM y OutputHead
        self.output_dropout = nn.Dropout(p=config.dropout)

        # --- Stage 5: Output Head ---
        # Bi-LSTM produce hidden_size*2 (forward + backward)
        # Usamos solo el último timestep para proyectar al horizonte
        self.output_head = nn.Linear(
            config.lstm_hidden_size * 2,
            config.forecast_horizon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass completo del Transformer Bi-LSTM.

        Args:
            x: Tensor de entrada con shape (batch_size, sequence_length, n_features).

        Returns:
            Tensor de predicciones con shape (batch_size, forecast_horizon).
        """
        # Stage 1: (batch, seq_len, n_features) → (batch, seq_len, d_model)
        x = self.input_projection(x)

        # Stage 2: Sumar positional encoding
        x = self.positional_encoding(x)

        # Stage 3: Self-attention sobre la secuencia
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        x = self.transformer_encoder(x)

        # Stage 4: Bi-LSTM procesa la secuencia completa
        # (batch, seq_len, d_model) → (batch, seq_len, lstm_hidden*2)
        x, _ = self.bilstm(x)

        # Tomar solo el último timestep (resumen de toda la secuencia)
        # (batch, seq_len, lstm_hidden*2) → (batch, lstm_hidden*2)
        x = x[:, -1, :]

        # Dropout antes de la proyección final
        x = self.output_dropout(x)

        # Stage 5: Proyección al horizonte de pronóstico
        # (batch, lstm_hidden*2) → (batch, forecast_horizon)
        x = self.output_head(x)

        return x

    def count_parameters(self) -> int:
        """Cuenta el número total de parámetros entrenables.

        Útil para logging estructurado y verificación de la
        complejidad del modelo antes del entrenamiento.

        Returns:
            Número total de parámetros con requires_grad=True.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
