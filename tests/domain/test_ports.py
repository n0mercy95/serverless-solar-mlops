# ============================================================
# test_ports.py — Tests de Puertos Abstractos
# Serverless Solar MLOps | Sub-fase 1.1
# ============================================================

from typing import Dict, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from domain.ports.ports import (
    CheckpointPort,
    DataPort,
    MetricsLoggerPort,
    ModelRepositoryPort,
)


class TestDataPort:
    """Tests para verificar que DataPort es una interfaz abstracta correcta."""

    def test_cannot_instantiate_directly(self) -> None:
        """No se puede instanciar un ABC directamente."""
        with pytest.raises(TypeError, match="abstract"):
            DataPort()  # type: ignore[abstract]

    def test_concrete_implementation_works(self) -> None:
        """Una implementación concreta que define todos los métodos debe funcionar."""

        class MockDataPort(DataPort):
            def load_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
                return torch.zeros(10, 24, 3), torch.zeros(10, 6)

            def load_validation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
                return torch.zeros(5, 24, 3), torch.zeros(5, 6)

        port = MockDataPort()
        X, y = port.load_training_data()
        assert X.shape == (10, 24, 3)
        assert y.shape == (10, 6)

    def test_partial_implementation_raises(self) -> None:
        """Implementación parcial (falta un método) debe fallar."""

        class PartialDataPort(DataPort):
            def load_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
                return torch.zeros(10, 24, 3), torch.zeros(10, 6)

        with pytest.raises(TypeError, match="abstract"):
            PartialDataPort()  # type: ignore[abstract]


class TestModelRepositoryPort:
    """Tests para verificar que ModelRepositoryPort es una interfaz abstracta correcta."""

    def test_cannot_instantiate_directly(self) -> None:
        """No se puede instanciar un ABC directamente."""
        with pytest.raises(TypeError, match="abstract"):
            ModelRepositoryPort()  # type: ignore[abstract]

    def test_concrete_implementation_works(self) -> None:
        """Una implementación concreta que define todos los métodos debe funcionar."""

        class MockModelRepo(ModelRepositoryPort):
            def save_model(self, model: nn.Module, metadata: Dict[str, object]) -> str:
                return "/mock/path/model.pt"

            def load_model(self, model_path: str) -> nn.Module:
                return nn.Linear(10, 1)

        port = MockModelRepo()
        path = port.save_model(nn.Linear(10, 1), {"epoch": 10})
        assert isinstance(path, str)


class TestCheckpointPort:
    """Tests para verificar que CheckpointPort es una interfaz abstracta correcta."""

    def test_cannot_instantiate_directly(self) -> None:
        """No se puede instanciar un ABC directamente."""
        with pytest.raises(TypeError, match="abstract"):
            CheckpointPort()  # type: ignore[abstract]

    def test_concrete_implementation_works(self) -> None:
        """Una implementación concreta que define todos los métodos debe funcionar."""

        class MockCheckpoint(CheckpointPort):
            def save_checkpoint(self, state: Dict[str, object], epoch: int) -> str:
                return f"/mock/checkpoint_epoch_{epoch}.pt"

            def load_latest_checkpoint(self) -> Optional[Dict[str, object]]:
                return None

        port = MockCheckpoint()
        path = port.save_checkpoint({"loss": 0.1}, epoch=5)
        assert "epoch_5" in path
        assert port.load_latest_checkpoint() is None


class TestMetricsLoggerPort:
    """Tests para verificar que MetricsLoggerPort es una interfaz abstracta correcta."""

    def test_cannot_instantiate_directly(self) -> None:
        """No se puede instanciar un ABC directamente."""
        with pytest.raises(TypeError, match="abstract"):
            MetricsLoggerPort()  # type: ignore[abstract]

    def test_concrete_implementation_works(self) -> None:
        """Una implementación concreta que define todos los métodos debe funcionar."""

        class MockLogger(MetricsLoggerPort):
            def __init__(self) -> None:
                self.logged: list = []

            def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
                self.logged.append({"epoch": epoch, **metrics})

            def log_training_complete(self, final_metrics: Dict[str, float]) -> None:
                self.logged.append({"complete": True, **final_metrics})

        logger = MockLogger()
        logger.log_epoch_metrics(1, {"train_loss": 0.5, "val_mae": 0.3})
        logger.log_training_complete({"best_val_mae": 0.1})
        assert len(logger.logged) == 2
        assert logger.logged[0]["epoch"] == 1
        assert logger.logged[1]["complete"] is True
