# ============================================================
# test_adapter_config.py — Tests de Configuración de Adaptadores
# Serverless Solar MLOps | Sub-fase 2.1
# ============================================================

import os

import pytest

from adapters.adapter_config import BigQueryConfig, StorageConfig


class TestBigQueryConfig:
    """Tests de validación Pydantic para BigQueryConfig."""

    def test_valid_config_with_explicit_values(self):
        """Config con valores explícitos se crea correctamente."""
        config = BigQueryConfig(
            project_id="my-gcp-project",
            dataset="gold_layer",
            table="solar_timeseries",
        )
        assert config.project_id == "my-gcp-project"
        assert config.dataset == "gold_layer"
        assert config.table == "solar_timeseries"

    def test_default_feature_columns_count(self):
        """Los feature_columns por defecto contienen 7 variables."""
        config = BigQueryConfig(project_id="test-project")
        assert len(config.feature_columns) == 7
        assert "power_output" in config.feature_columns
        assert "ghi" in config.feature_columns

    def test_default_target_column(self):
        """El target_column por defecto es power_output."""
        config = BigQueryConfig(project_id="test-project")
        assert config.target_column == "power_output"

    def test_default_split_ratio(self):
        """El split_ratio por defecto es 0.8."""
        config = BigQueryConfig(project_id="test-project")
        assert config.split_ratio == 0.8

    def test_empty_project_id_raises_error(self):
        """Un project_id vacío lanza ValueError."""
        with pytest.raises(ValueError, match="GCP_PROJECT_ID"):
            BigQueryConfig(project_id="")

    def test_whitespace_project_id_raises_error(self):
        """Un project_id con solo espacios lanza ValueError."""
        with pytest.raises(ValueError, match="GCP_PROJECT_ID"):
            BigQueryConfig(project_id="   ")

    def test_project_id_is_stripped(self):
        """Los espacios en project_id se eliminan."""
        config = BigQueryConfig(project_id="  my-project  ")
        assert config.project_id == "my-project"

    def test_split_ratio_zero_raises_error(self):
        """Un split_ratio de 0.0 lanza error de validación."""
        with pytest.raises(ValueError):
            BigQueryConfig(project_id="test", split_ratio=0.0)

    def test_split_ratio_one_raises_error(self):
        """Un split_ratio de 1.0 lanza error de validación."""
        with pytest.raises(ValueError):
            BigQueryConfig(project_id="test", split_ratio=1.0)

    def test_split_ratio_negative_raises_error(self):
        """Un split_ratio negativo lanza error de validación."""
        with pytest.raises(ValueError):
            BigQueryConfig(project_id="test", split_ratio=-0.5)

    def test_target_not_in_features_raises_error(self):
        """Un target_column que no está en feature_columns lanza ValueError."""
        with pytest.raises(ValueError, match="target_column"):
            BigQueryConfig(
                project_id="test",
                feature_columns=["ghi", "temperature"],
                target_column="power_output",
            )

    def test_custom_feature_columns(self):
        """Se pueden especificar feature_columns personalizadas."""
        config = BigQueryConfig(
            project_id="test",
            feature_columns=["ghi", "temperature", "power_output"],
            target_column="power_output",
        )
        assert len(config.feature_columns) == 3

    def test_reads_from_env_variables(self, monkeypatch):
        """Lee valores por defecto desde variables de entorno."""
        monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
        monkeypatch.setenv("BQ_DATASET_GOLD", "env-dataset")
        monkeypatch.setenv("BQ_TABLE_TIMESERIES", "env-table")
        config = BigQueryConfig()
        assert config.project_id == "env-project"
        assert config.dataset == "env-dataset"
        assert config.table == "env-table"


class TestStorageConfig:
    """Tests de validación Pydantic para StorageConfig."""

    def test_default_model_dir(self):
        """El model_dir por defecto apunta a /gcs/model-output."""
        config = StorageConfig()
        assert config.model_dir == "/gcs/model-output"

    def test_default_checkpoint_dir(self):
        """El checkpoint_dir por defecto apunta a /gcs/checkpoints."""
        config = StorageConfig()
        assert config.checkpoint_dir == "/gcs/checkpoints"

    def test_default_filenames(self):
        """Los nombres de archivo por defecto son model.pt y metadata.json."""
        config = StorageConfig()
        assert config.model_filename == "model.pt"
        assert config.metadata_filename == "metadata.json"

    def test_custom_paths(self):
        """Se pueden configurar rutas personalizadas."""
        config = StorageConfig(
            model_dir="/custom/models",
            checkpoint_dir="/custom/checkpoints",
        )
        assert config.model_dir == "/custom/models"
        assert config.checkpoint_dir == "/custom/checkpoints"

    def test_empty_model_dir_raises_error(self):
        """Un model_dir vacío lanza ValueError."""
        with pytest.raises(ValueError, match="directorio"):
            StorageConfig(model_dir="")

    def test_empty_checkpoint_dir_raises_error(self):
        """Un checkpoint_dir vacío lanza ValueError."""
        with pytest.raises(ValueError, match="directorio"):
            StorageConfig(checkpoint_dir="")

    def test_reads_from_env_variables(self, monkeypatch):
        """Lee rutas desde las variables de entorno de Vertex AI."""
        monkeypatch.setenv("AIP_MODEL_DIR", "/vertex/model")
        monkeypatch.setenv("AIP_CHECKPOINT_DIR", "/vertex/checkpoints")
        config = StorageConfig()
        assert config.model_dir == "/vertex/model"
        assert config.checkpoint_dir == "/vertex/checkpoints"

    def test_paths_are_stripped(self):
        """Los espacios en las rutas se eliminan."""
        config = StorageConfig(
            model_dir="  /path/with/spaces  ",
            checkpoint_dir="  /another/path  ",
        )
        assert config.model_dir == "/path/with/spaces"
        assert config.checkpoint_dir == "/another/path"
