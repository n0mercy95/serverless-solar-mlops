# ============================================================
# test_strategies.py — Tests para Estrategias de Pérdida
# Serverless Solar MLOps | Sub-fase 1.2
# ============================================================

import pytest
import torch

from domain.strategies.loss_strategies import (
    LossStrategy,
    MAELossStrategy,
    MSELossStrategy,
    RMSELossStrategy,
)


class TestLossStrategyABC:
    """Verifica que LossStrategy se comporte como una clase abstracta pura."""

    def test_cannot_instantiate_directly(self) -> None:
        """No se puede instanciar el ABC directamente."""
        with pytest.raises(TypeError, match="abstract"):
            LossStrategy()  # type: ignore[abstract]

    def test_concrete_implementation_works(self) -> None:
        """Una implementación concreta simple debe poder instanciarse."""

        class DummyLoss(LossStrategy):
            def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return torch.tensor(0.0)

            @property
            def name(self) -> str:
                return "dummy"

        loss = DummyLoss()
        assert loss.name == "dummy"
        assert loss.compute(torch.zeros(1), torch.zeros(1)) == 0.0


class TestMAELossStrategy:
    """Tests unitarios para MAELossStrategy."""

    def test_name(self) -> None:
        strategy = MAELossStrategy()
        assert strategy.name == "mae"

    def test_correctness(self) -> None:
        strategy = MAELossStrategy()
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[2.0, 2.0], [1.0, 6.0]])
        # Diferencias absolutas: |1-2| + |2-2| + |3-1| + |4-6| = 1 + 0 + 2 + 2 = 5
        # MAE: 5 / 4 = 1.25
        loss = strategy.compute(pred, target)
        assert torch.isclose(loss, torch.tensor(1.25))

    def test_gradient_propagation(self) -> None:
        strategy = MAELossStrategy()
        pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[2.0, 4.0]])
        loss = strategy.compute(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0.0)


class TestMSELossStrategy:
    """Tests unitarios para MSELossStrategy."""

    def test_name(self) -> None:
        strategy = MSELossStrategy()
        assert strategy.name == "mse"

    def test_correctness(self) -> None:
        strategy = MSELossStrategy()
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[2.0, 2.0], [1.0, 6.0]])
        # Diferencias al cuadrado: (1-2)^2 + (2-2)^2 + (3-1)^2 + (4-6)^2 = 1 + 0 + 4 + 4 = 9
        # MSE: 9 / 4 = 2.25
        loss = strategy.compute(pred, target)
        assert torch.isclose(loss, torch.tensor(2.25))

    def test_gradient_propagation(self) -> None:
        strategy = MSELossStrategy()
        pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[2.0, 4.0]])
        loss = strategy.compute(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0.0)


class TestRMSELossStrategy:
    """Tests unitarios para RMSELossStrategy."""

    def test_name(self) -> None:
        strategy = RMSELossStrategy()
        assert strategy.name == "rmse"

    def test_correctness(self) -> None:
        strategy = RMSELossStrategy(eps=0.0)
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[2.0, 2.0], [1.0, 6.0]])
        # MSE = 2.25
        # RMSE = sqrt(2.25) = 1.5
        loss = strategy.compute(pred, target)
        assert torch.isclose(loss, torch.tensor(1.5))

    def test_stability_with_zero_error(self) -> None:
        """RMSE debe ser estable (no lanzar NaN) cuando el error es exactamente cero."""
        strategy = RMSELossStrategy(eps=1e-8)
        pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0]])
        loss = strategy.compute(pred, target)
        # Sin epsilon, la derivada de sqrt(0) sería NaN/infinito en PyTorch.
        loss.backward()
        assert not torch.isnan(loss)
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_gradient_propagation(self) -> None:
        strategy = RMSELossStrategy()
        pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[2.0, 4.0]])
        loss = strategy.compute(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0.0)
