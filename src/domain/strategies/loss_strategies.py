# ============================================================
# loss_strategies.py — Estrategias de Pérdida y Evaluación
# Serverless Solar MLOps | Sub-fase 1.2
# ============================================================
# Patrón de Diseño Strategy (Gang of Four).
# Abstrae la computación de las funciones de pérdida matemática,
# permitiendo al motor de entrenamiento optimizar o evaluar
# usando diferentes criterios sin alterar el bucle principal.
# ============================================================

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class LossStrategy(ABC):
    """Interfaz abstracta para el cálculo de pérdidas y métricas.

    Define el contrato para cualquier función de pérdida matemática
    que compare las predicciones del modelo con las etiquetas reales (targets).
    """

    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcula el valor de la pérdida.

        Args:
            predictions: Tensor con las predicciones del modelo (batch_size, forecast_horizon).
            targets: Tensor con los valores reales (batch_size, forecast_horizon).

        Returns:
            Tensor escalar (0-dimensional) con el valor de la pérdida.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Retorna el nombre identificador de la estrategia."""


class MAELossStrategy(LossStrategy):
    """Estrategia para Mean Absolute Error (MAE / L1 Loss).

    Mide el promedio de las diferencias absolutas entre predicciones y targets.
    Es más robusto frente a valores atípicos (outliers).
    """

    def __init__(self) -> None:
        self._loss_fn = nn.L1Loss()

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(predictions, targets)

    @property
    def name(self) -> str:
        return "mae"


class MSELossStrategy(LossStrategy):
    """Estrategia para Mean Squared Error (MSE / L2 Loss).

    Mide el promedio de los errores al cuadrado. Es diferenciable y penaliza
    fuertemente los errores grandes, siendo ideal para el backpropagation.
    """

    def __init__(self) -> None:
        self._loss_fn = nn.MSELoss()

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(predictions, targets)

    @property
    def name(self) -> str:
        return "mse"


class RMSELossStrategy(LossStrategy):
    """Estrategia para Root Mean Squared Error (RMSE).

    Calcula la raíz cuadrada del Mean Squared Error. Mantiene la misma
    escala física que la variable objetivo original (ej. kW/MW).

    Fórmula: √(MSE(y_pred, y_true) + epsilon)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """
        Args:
            eps: Pequeño valor para evitar raíces de cero o derivadas inestables.
        """
        self._mse_fn = nn.MSELoss()
        self._eps = eps

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = self._mse_fn(predictions, targets)
        return torch.sqrt(mse + self._eps)

    @property
    def name(self) -> str:
        return "rmse"
