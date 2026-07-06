# ============================================================
# ports.py — Puertos Abstractos (Arquitectura Hexagonal)
# Serverless Solar MLOps | Sub-fase 1.1
# ============================================================
# Define las interfaces (contratos) que el dominio expone
# hacia la infraestructura. Los adaptadores concretos (BigQuery,
# Vertex AI Storage, etc.) implementarán estos puertos en la
# Fase 2, cumpliendo el Principio de Inversión de Dependencias.
#
# NOTA: Los tipos usan torch.Tensor porque PyTorch es una
#       dependencia del dominio matemático, NO de infraestructura.
# ============================================================

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class DataPort(ABC):
    """Puerto de entrada: adquisición de datos de series de tiempo.

    Define el contrato para cargar datos de entrenamiento y validación
    desde cualquier fuente (BigQuery, CSV local, mock para tests).

    El adaptador concreto (Sub-fase 2.1) se encargará de:
    - Ejecutar la query SQL a la Capa Oro de BigQuery
    - Transformar el resultado a tensores PyTorch
    - Aplicar windowing (ventanas deslizantes) sobre la serie temporal
    """

    @abstractmethod
    def load_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carga los datos de entrenamiento.

        Returns:
            Tupla (X_train, y_train) donde:
            - X_train: Tensor shape (n_samples, sequence_length, n_features)
            - y_train: Tensor shape (n_samples, forecast_horizon)
        """

    @abstractmethod
    def load_validation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carga los datos de validación.

        Returns:
            Tupla (X_val, y_val) con las mismas shapes que load_training_data.
        """


class ModelRepositoryPort(ABC):
    """Puerto de salida: persistencia de artefactos del modelo.

    Define el contrato para guardar/cargar el modelo entrenado.
    El adaptador concreto (Sub-fase 2.1) implementará la lógica
    de exportar pesos a Vertex AI Artifact Registry / GCS.
    """

    @abstractmethod
    def save_model(
        self, model: nn.Module, metadata: Dict[str, object]
    ) -> str:
        """Persiste el modelo entrenado y sus metadatos.

        Args:
            model: Instancia del modelo PyTorch entrenado.
            metadata: Diccionario con métricas, config, timestamp, etc.

        Returns:
            Ruta o URI del artefacto guardado.
        """

    @abstractmethod
    def load_model(self, model_path: str) -> nn.Module:
        """Carga un modelo previamente guardado.

        Args:
            model_path: Ruta o URI del artefacto a cargar.

        Returns:
            Instancia del modelo PyTorch con pesos restaurados.
        """


class CheckpointPort(ABC):
    """Puerto de salida: gestión de checkpoints para tolerancia a fallos.

    Los checkpoints permiten reanudar el entrenamiento si el job
    de Vertex AI falla o es interrumpido (preemptible VMs).
    El adaptador concreto escribirá a AIP_CHECKPOINT_DIR.
    """

    @abstractmethod
    def save_checkpoint(
        self, state: Dict[str, object], epoch: int
    ) -> str:
        """Guarda un checkpoint del estado de entrenamiento.

        Args:
            state: Diccionario con model.state_dict(), optimizer.state_dict(),
                   epoch actual, métricas, etc.
            epoch: Número de época (para nombrar el checkpoint).

        Returns:
            Ruta del checkpoint guardado.
        """

    @abstractmethod
    def load_latest_checkpoint(self) -> Optional[Dict[str, object]]:
        """Carga el checkpoint más reciente disponible.

        Returns:
            Diccionario con el estado guardado, o None si no existe
            ningún checkpoint previo.
        """


class MetricsLoggerPort(ABC):
    """Puerto de salida: logging de métricas de entrenamiento.

    Define el contrato para registrar métricas durante el
    entrenamiento. El adaptador concreto (Sub-fase 2.2) implementará
    el CloudStructuredLogFormatter para emitir JSON a Cloud Logging.
    """

    @abstractmethod
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Registra las métricas de una época completada.

        Args:
            epoch: Número de la época actual.
            metrics: Diccionario con métricas (ej. {"train_loss": 0.05,
                     "val_mae": 0.03, "val_rmse": 0.04}).
        """

    @abstractmethod
    def log_training_complete(self, final_metrics: Dict[str, float]) -> None:
        """Registra la finalización exitosa del entrenamiento.

        Args:
            final_metrics: Diccionario con las métricas finales del
                          mejor modelo (ej. best_val_mae, total_epochs).
        """
