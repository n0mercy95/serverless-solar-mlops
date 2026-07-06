# ============================================================
# config.py — Esquemas de Configuración del Modelo
# Serverless Solar MLOps | Sub-fase 1.1
# ============================================================
# Validación estricta de hiperparámetros mediante Pydantic.
# Los defaults provienen del .env.example del proyecto.
# ============================================================

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Configuración validada del Transformer Bi-LSTM.

    Encapsula todos los hiperparámetros arquitectónicos del modelo,
    aplicando restricciones matemáticas (ej. d_model divisible por n_heads)
    para prevenir errores silenciosos en tiempo de construcción.
    """

    n_features: int = Field(
        default=7,
        ge=1,
        description="Número de features de entrada en la serie temporal",
    )
    d_model: int = Field(
        default=128,
        ge=1,
        description="Dimensión del espacio de embedding del Transformer",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        description="Número de cabezas de atención en el Transformer Encoder",
    )
    n_encoder_layers: int = Field(
        default=3,
        ge=1,
        description="Número de capas del Transformer Encoder",
    )
    lstm_hidden_size: int = Field(
        default=256,
        ge=1,
        description="Tamaño del estado oculto de cada dirección del Bi-LSTM",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Tasa de dropout para regularización",
    )
    sequence_length: int = Field(
        default=168,
        ge=1,
        description="Longitud de la ventana de entrada (ej. 168 = 7 días × 24h)",
    )
    forecast_horizon: int = Field(
        default=24,
        ge=1,
        description="Número de pasos temporales a predecir (ej. 24h)",
    )

    @model_validator(mode="after")
    def validate_d_model_divisible_by_n_heads(self) -> "ModelConfig":
        """Garantiza que d_model sea divisible por n_heads.

        Esta es una restricción matemática del mecanismo de Multi-Head
        Attention: cada cabeza opera sobre d_model // n_heads dimensiones.
        """
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) debe ser divisible por "
                f"n_heads ({self.n_heads}). "
                f"Residuo: {self.d_model % self.n_heads}"
            )
        return self


class TrainingConfig(BaseModel):
    """Configuración validada de hiperparámetros de entrenamiento.

    Separa los parámetros de entrenamiento de los arquitectónicos
    para permitir experimentación independiente.
    """

    epochs: int = Field(
        default=100,
        ge=1,
        description="Número total de épocas de entrenamiento",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Tamaño del mini-batch para SGD",
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Tasa de aprendizaje del optimizador",
    )
