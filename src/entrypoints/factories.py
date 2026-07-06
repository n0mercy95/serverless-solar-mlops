# ============================================================
# factories.py — Fábricas de Creación Dinámica (GoF Factory)
# Serverless Solar MLOps | Sub-fase 1.2
# ============================================================
# Patrón de Diseño Factory Method / Registry (Gang of Four).
# Encapsula la creación e instanciación de modelos matemáticos
# y estrategias de pérdida, independizando los entrypoints de las
# implementaciones concretas y facilitando la experimentación.
# ============================================================

from typing import Dict, List, Type
import torch.nn as nn

from domain.models.config import ModelConfig
from domain.models.transformer_bilstm import TransformerBiLSTM
from domain.strategies.loss_strategies import (
    LossStrategy,
    MAELossStrategy,
    MSELossStrategy,
    RMSELossStrategy,
)


class ModelFactory:
    """Fábrica creacional con registro dinámico de modelos PyTorch.

    Permite instanciar diferentes arquitecturas profundas a partir de
    su tipo y configuración, promoviendo el Principio de Abierto/Cerrado (OCP).
    """

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module]) -> None:
        """Registra una clase de modelo en la fábrica.

        Args:
            name: Nombre identificador único del modelo (ej. 'transformer_bilstm').
            model_class: Clase del modelo (debe ser subclase de nn.Module).
        """
        cls._registry[name.lower().strip()] = model_class

    @classmethod
    def create(cls, model_type: str, config: ModelConfig) -> nn.Module:
        """Instancia dinámicamente un modelo a partir de su tipo y configuración.

        Args:
            model_type: Identificador del modelo (ej. 'transformer_bilstm').
            config: Instancia validada de ModelConfig.

        Returns:
            Instancia concreta del modelo (nn.Module).

        Raises:
            ValueError: Si el model_type solicitado no se encuentra registrado.
        """
        key = model_type.lower().strip()
        if key not in cls._registry:
            raise ValueError(
                f"El modelo '{model_type}' no está registrado en la fábrica. "
                f"Modelos registrados: {list(cls._registry.keys())}"
            )
        return cls._registry[key](config)


class LossFactory:
    """Fábrica estática para la creación de estrategias de pérdida (LossStrategy)."""

    _strategies: Dict[str, Type[LossStrategy]] = {
        "mae": MAELossStrategy,
        "mse": MSELossStrategy,
        "rmse": RMSELossStrategy,
    }

    @classmethod
    def create(cls, loss_name: str) -> LossStrategy:
        """Instancia la estrategia de pérdida correspondiente.

        Args:
            loss_name: Nombre de la pérdida ('mae', 'mse', 'rmse').

        Returns:
            Instancia concreta de LossStrategy.

        Raises:
            ValueError: Si el loss_name solicitado no está soportado.
        """
        key = loss_name.lower().strip()
        if key not in cls._strategies:
            raise ValueError(
                f"La estrategia de pérdida '{loss_name}' no está soportada. "
                f"Opciones válidas: {list(cls._strategies.keys())}"
            )
        return cls._strategies[key]()


# Autoregistro del modelo base TransformerBiLSTM al importar la fábrica
ModelFactory.register("transformer_bilstm", TransformerBiLSTM)
