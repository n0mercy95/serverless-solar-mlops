# ============================================================
# test_model_adapters.py — Tests de Adaptadores de Modelo/Checkpoint
# Serverless Solar MLOps | Sub-fase 2.1
# ============================================================
# Tests de roundtrip (save → load) usando tmp_path de pytest.
# Verifican la serialización correcta de pesos, metadatos,
# y checkpoints sin necesidad de un sistema de archivos real.
# ============================================================

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from adapters.model_adapters import (
    VertexCheckpointAdapter,
    VertexModelRepositoryAdapter,
    _make_json_serializable,
)
from domain.models.config import ModelConfig
from domain.models.transformer_bilstm import TransformerBiLSTM


# ---- Fixtures ----


@pytest.fixture
def small_model():
    """Modelo TransformerBiLSTM pequeño para tests rápidos."""
    config = ModelConfig(
        n_features=3,
        d_model=16,
        n_heads=4,
        n_encoder_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
        sequence_length=24,
        forecast_horizon=6,
    )
    return TransformerBiLSTM(config)


@pytest.fixture
def model_adapter(tmp_path):
    """VertexModelRepositoryAdapter apuntando a tmp_path."""
    return VertexModelRepositoryAdapter(model_dir=str(tmp_path / "model-output"))


@pytest.fixture
def checkpoint_adapter(tmp_path):
    """VertexCheckpointAdapter apuntando a tmp_path."""
    return VertexCheckpointAdapter(checkpoint_dir=str(tmp_path / "checkpoints"))


@pytest.fixture
def sample_metadata():
    """Metadatos de ejemplo para guardar junto al modelo."""
    return {
        "model_type": "transformer_bilstm",
        "best_val_mae": 0.0342,
        "best_val_rmse": 0.0478,
        "total_epochs": 50,
        "early_stop_epoch": 42,
        "config": {
            "d_model": 16,
            "n_heads": 4,
            "n_encoder_layers": 1,
        },
    }


# ---- Tests de VertexModelRepositoryAdapter ----


class TestVertexModelRepositoryAdapter:
    """Tests del adaptador de persistencia del modelo."""

    def test_save_creates_model_file(self, model_adapter, small_model, sample_metadata):
        """save_model crea el archivo model.pt."""
        model_adapter.save_model(small_model, sample_metadata)
        model_path = Path(model_adapter._model_dir) / "model.pt"
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_save_creates_metadata_file(
        self, model_adapter, small_model, sample_metadata
    ):
        """save_model crea el archivo metadata.json."""
        model_adapter.save_model(small_model, sample_metadata)
        metadata_path = Path(model_adapter._model_dir) / "metadata.json"
        assert metadata_path.exists()

    def test_save_returns_directory_path(
        self, model_adapter, small_model, sample_metadata
    ):
        """save_model retorna la ruta del directorio de salida."""
        result = model_adapter.save_model(small_model, sample_metadata)
        assert result == str(model_adapter._model_dir)

    def test_roundtrip_weights_are_identical(
        self, model_adapter, small_model, sample_metadata
    ):
        """Los pesos guardados y cargados son idénticos (roundtrip)."""
        model_adapter.save_model(small_model, sample_metadata)
        loaded_state = model_adapter.load_model(str(model_adapter._model_dir))

        original_state = small_model.state_dict()
        for key in original_state:
            assert key in loaded_state
            assert torch.equal(original_state[key], loaded_state[key]), (
                f"Tensor '{key}' no coincide tras el roundtrip"
            )

    def test_load_metadata_roundtrip(
        self, model_adapter, small_model, sample_metadata
    ):
        """Los metadatos guardados y cargados son idénticos."""
        model_adapter.save_model(small_model, sample_metadata)
        loaded_metadata = model_adapter.load_metadata(str(model_adapter._model_dir))
        assert loaded_metadata["model_type"] == "transformer_bilstm"
        assert loaded_metadata["best_val_mae"] == pytest.approx(0.0342)
        assert loaded_metadata["total_epochs"] == 50

    def test_metadata_json_is_valid(
        self, model_adapter, small_model, sample_metadata
    ):
        """El archivo metadata.json contiene JSON válido y parseable."""
        model_adapter.save_model(small_model, sample_metadata)
        metadata_path = Path(model_adapter._model_dir) / "metadata.json"
        with open(metadata_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "config" in data

    def test_save_creates_directory_if_not_exists(
        self, tmp_path, small_model, sample_metadata
    ):
        """save_model crea el directorio de salida si no existe."""
        deep_path = tmp_path / "nested" / "deep" / "model"
        adapter = VertexModelRepositoryAdapter(model_dir=str(deep_path))
        adapter.save_model(small_model, sample_metadata)
        assert deep_path.exists()

    def test_load_nonexistent_model_raises_error(self, model_adapter):
        """Cargar desde un directorio sin model.pt lanza RuntimeError."""
        with pytest.raises(RuntimeError, match="Error al cargar modelo"):
            model_adapter.load_model("/nonexistent/path")

    def test_load_nonexistent_metadata_raises_error(self, model_adapter, tmp_path):
        """Cargar metadatos inexistentes lanza RuntimeError."""
        empty_dir = tmp_path / "empty-model"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError, match="Error al cargar metadatos"):
            model_adapter.load_metadata(str(empty_dir))

    def test_metadata_with_tensor_values(
        self, model_adapter, small_model
    ):
        """Los metadatos con tensores PyTorch se serializan correctamente."""
        metadata_with_tensors = {
            "loss": torch.tensor(0.05),
            "history": torch.tensor([0.1, 0.08, 0.06]),
        }
        model_adapter.save_model(small_model, metadata_with_tensors)
        loaded = model_adapter.load_metadata(str(model_adapter._model_dir))
        assert loaded["loss"] == pytest.approx(0.05)
        assert loaded["history"] == pytest.approx([0.1, 0.08, 0.06])


# ---- Tests de VertexCheckpointAdapter ----


class TestVertexCheckpointAdapter:
    """Tests del adaptador de checkpoints para tolerancia a fallos."""

    def test_save_creates_checkpoint_file(self, checkpoint_adapter):
        """save_checkpoint crea el archivo con el nombre correcto."""
        state = {"epoch": 5, "loss": 0.03}
        path = checkpoint_adapter.save_checkpoint(state, epoch=5)
        assert Path(path).exists()
        assert "checkpoint_epoch_005.pt" in path

    def test_save_checkpoint_returns_path(self, checkpoint_adapter):
        """save_checkpoint retorna la ruta completa del archivo."""
        state = {"epoch": 1}
        path = checkpoint_adapter.save_checkpoint(state, epoch=1)
        assert path.endswith("checkpoint_epoch_001.pt")

    def test_save_creates_directory_if_not_exists(self, tmp_path):
        """save_checkpoint crea el directorio si no existe."""
        deep_path = tmp_path / "nested" / "checkpoints"
        adapter = VertexCheckpointAdapter(checkpoint_dir=str(deep_path))
        adapter.save_checkpoint({"epoch": 1}, epoch=1)
        assert deep_path.exists()

    def test_load_latest_returns_most_recent(self, checkpoint_adapter):
        """load_latest_checkpoint retorna el checkpoint con epoch más alto."""
        for epoch in [1, 3, 2, 5, 4]:
            state = {"epoch": epoch, "loss": 1.0 / epoch}
            checkpoint_adapter.save_checkpoint(state, epoch=epoch)

        loaded = checkpoint_adapter.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["epoch"] == 5

    def test_load_latest_returns_none_when_no_checkpoints(self, checkpoint_adapter):
        """load_latest_checkpoint retorna None si no hay checkpoints."""
        result = checkpoint_adapter.load_latest_checkpoint()
        assert result is None

    def test_load_latest_returns_none_when_dir_not_exists(self, tmp_path):
        """load_latest_checkpoint retorna None si el directorio no existe."""
        adapter = VertexCheckpointAdapter(
            checkpoint_dir=str(tmp_path / "nonexistent")
        )
        result = adapter.load_latest_checkpoint()
        assert result is None

    def test_checkpoint_roundtrip_state_dict(self, checkpoint_adapter, small_model):
        """Roundtrip de state_dict del modelo a través de checkpoint."""
        original_state = small_model.state_dict()
        checkpoint_state = {
            "epoch": 10,
            "model_state_dict": original_state,
            "optimizer_state_dict": {"lr": 0.001},
            "val_mae": 0.03,
        }
        checkpoint_adapter.save_checkpoint(checkpoint_state, epoch=10)

        loaded = checkpoint_adapter.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["epoch"] == 10
        assert loaded["val_mae"] == pytest.approx(0.03)

        # Verificar pesos del modelo
        for key in original_state:
            assert torch.equal(
                original_state[key], loaded["model_state_dict"][key]
            ), f"Tensor '{key}' no coincide en checkpoint roundtrip"

    def test_multiple_checkpoints_are_persisted(self, checkpoint_adapter):
        """Múltiples checkpoints coexisten en el directorio."""
        for epoch in range(1, 6):
            checkpoint_adapter.save_checkpoint({"epoch": epoch}, epoch=epoch)

        checkpoint_dir = Path(checkpoint_adapter._checkpoint_dir)
        pt_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(pt_files) == 5

    def test_ignores_non_checkpoint_files(self, checkpoint_adapter, tmp_path):
        """load_latest_checkpoint ignora archivos que no son checkpoints."""
        # Guardar un checkpoint válido
        checkpoint_adapter.save_checkpoint({"epoch": 3}, epoch=3)

        # Crear archivos que no son checkpoints
        checkpoint_dir = Path(checkpoint_adapter._checkpoint_dir)
        (checkpoint_dir / "random_file.pt").write_bytes(b"not a checkpoint")
        (checkpoint_dir / "notes.txt").write_text("some notes")

        loaded = checkpoint_adapter.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["epoch"] == 3

    def test_epoch_zero_naming(self, checkpoint_adapter):
        """El epoch 0 genera un nombre de archivo válido."""
        path = checkpoint_adapter.save_checkpoint({"epoch": 0}, epoch=0)
        assert "checkpoint_epoch_000.pt" in path


# ---- Tests de _make_json_serializable ----


class TestMakeJsonSerializable:
    """Tests de la función utilitaria de serialización JSON."""

    def test_tensor_scalar(self):
        """Convierte un tensor escalar a float."""
        result = _make_json_serializable(torch.tensor(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_tensor_1d(self):
        """Convierte un tensor 1D a lista."""
        result = _make_json_serializable(torch.tensor([1.0, 2.0, 3.0]))
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_nested_dict_with_tensors(self):
        """Convierte recursivamente dicts con tensores."""
        data = {
            "loss": torch.tensor(0.05),
            "config": {"lr": 0.001, "epochs": 100},
        }
        result = _make_json_serializable(data)
        assert result["loss"] == pytest.approx(0.05)
        assert result["config"]["lr"] == 0.001

    def test_numpy_scalar(self):
        """Convierte escalares numpy a tipos Python nativos."""
        import numpy as np
        result = _make_json_serializable(np.float64(3.14))
        assert isinstance(result, float)

    def test_numpy_array(self):
        """Convierte arrays numpy a listas."""
        import numpy as np
        result = _make_json_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_plain_dict_unchanged(self):
        """Un dict con tipos nativos se retorna sin cambios."""
        data = {"a": 1, "b": "hello", "c": [1, 2]}
        result = _make_json_serializable(data)
        assert result == data

    def test_result_is_json_serializable(self):
        """El resultado puede ser serializado por json.dumps sin error."""
        import numpy as np
        data = {
            "tensor": torch.tensor([1.0, 2.0]),
            "np_val": np.float32(3.14),
            "nested": {"arr": np.array([4, 5])},
        }
        result = _make_json_serializable(data)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
