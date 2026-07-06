# ============================================================
# test_data_adapters.py — Tests del BigQueryTimeSeriesAdapter
# Serverless Solar MLOps | Sub-fase 2.1
# ============================================================
# Tests unitarios que mockean el SDK de BigQuery para validar
# la lógica de query, windowing, split y conversión a tensores.
# ============================================================

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from adapters.data_adapters import BigQueryTimeSeriesAdapter


# ---- Fixtures ----


@pytest.fixture
def feature_columns():
    """Columnas de features del esquema estándar fotovoltaico."""
    return [
        "ghi", "dni", "dhi", "temperature",
        "humidity", "wind_speed", "power_output",
    ]


@pytest.fixture
def mock_bq_client():
    """Cliente de BigQuery mockeado."""
    return MagicMock()


@pytest.fixture
def sample_dataframe(feature_columns):
    """DataFrame simulando la salida de BigQuery con 500 filas.

    Genera datos sintéticos con un patrón sinusoidal para simular
    una serie temporal fotovoltaica realista.
    """
    n_rows = 500
    timestamps = pd.date_range("2025-01-01", periods=n_rows, freq="h")
    data = {
        "timestamp": timestamps,
    }
    rng = np.random.default_rng(42)
    for col in feature_columns:
        data[col] = rng.random(n_rows).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture
def adapter_with_mock(mock_bq_client, feature_columns):
    """BigQueryTimeSeriesAdapter con cliente BigQuery mockeado."""
    return BigQueryTimeSeriesAdapter(
        project_id="test-project",
        dataset="gold_layer",
        table="solar_timeseries",
        feature_columns=feature_columns,
        target_column="power_output",
        timestamp_column="timestamp",
        sequence_length=24,
        forecast_horizon=6,
        split_ratio=0.8,
        client=mock_bq_client,
    )


# ---- Tests de Construcción de Query ----


class TestBuildQuery:
    """Tests del método _build_query."""

    def test_query_contains_table_reference(self, adapter_with_mock):
        """La query SQL referencia la tabla completa del proyecto."""
        query = adapter_with_mock._build_query()
        assert "test-project.gold_layer.solar_timeseries" in query

    def test_query_contains_all_feature_columns(
        self, adapter_with_mock, feature_columns
    ):
        """La query incluye todas las columnas de features."""
        query = adapter_with_mock._build_query()
        for col in feature_columns:
            assert col in query

    def test_query_contains_timestamp(self, adapter_with_mock):
        """La query incluye el timestamp y ordena ASC."""
        query = adapter_with_mock._build_query()
        assert "timestamp" in query
        assert "ORDER BY timestamp ASC" in query

    def test_query_is_select_statement(self, adapter_with_mock):
        """La query es un SELECT (no modifica datos)."""
        query = adapter_with_mock._build_query()
        assert query.strip().startswith("SELECT")


# ---- Tests de Windowing (Ventanas Deslizantes) ----


class TestCreateSequences:
    """Tests del método estático _create_sequences."""

    def test_output_shapes_are_correct(self):
        """Las shapes de X e y son correctas para los parámetros dados."""
        data = np.random.randn(100, 7).astype(np.float32)
        X, y = BigQueryTimeSeriesAdapter._create_sequences(
            data, target_idx=6, sequence_length=24, forecast_horizon=6
        )
        # n_windows = 100 - (24 + 6) + 1 = 71
        assert X.shape == (71, 24, 7)
        assert y.shape == (71, 6)

    def test_x_contains_correct_window(self):
        """Cada ventana X[i] contiene la porción correcta de datos."""
        data = np.arange(50).reshape(50, 1).astype(np.float32)
        X, _ = BigQueryTimeSeriesAdapter._create_sequences(
            data, target_idx=0, sequence_length=5, forecast_horizon=2
        )
        # Primera ventana: [0, 1, 2, 3, 4]
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3, 4])
        # Segunda ventana: [1, 2, 3, 4, 5]
        np.testing.assert_array_equal(X[1, :, 0], [1, 2, 3, 4, 5])

    def test_y_contains_correct_target(self):
        """Cada target y[i] contiene los valores futuros correctos."""
        data = np.arange(50).reshape(50, 1).astype(np.float32)
        _, y = BigQueryTimeSeriesAdapter._create_sequences(
            data, target_idx=0, sequence_length=5, forecast_horizon=2
        )
        # Primera ventana target: [5, 6]
        np.testing.assert_array_equal(y[0], [5, 6])
        # Segunda ventana target: [6, 7]
        np.testing.assert_array_equal(y[1], [6, 7])

    def test_multivariate_preserves_all_features(self):
        """El windowing preserva todas las features en X."""
        data = np.random.randn(50, 3).astype(np.float32)
        X, _ = BigQueryTimeSeriesAdapter._create_sequences(
            data, target_idx=2, sequence_length=10, forecast_horizon=3
        )
        assert X.shape[2] == 3  # Las 3 features se preservan

    def test_insufficient_data_raises_error(self):
        """Datos insuficientes para ni una ventana lanzan ValueError."""
        data = np.random.randn(5, 7).astype(np.float32)
        with pytest.raises(ValueError, match="Datos insuficientes"):
            BigQueryTimeSeriesAdapter._create_sequences(
                data, target_idx=6, sequence_length=24, forecast_horizon=6
            )

    def test_exact_minimum_data_produces_one_window(self):
        """Con exactamente seq_len + horizon datos, se produce 1 ventana."""
        seq_len, horizon = 10, 3
        data = np.random.randn(seq_len + horizon, 2).astype(np.float32)
        X, y = BigQueryTimeSeriesAdapter._create_sequences(
            data, target_idx=0, sequence_length=seq_len, forecast_horizon=horizon
        )
        assert X.shape[0] == 1
        assert y.shape[0] == 1

    def test_output_dtype_is_float32(self):
        """Los arrays de salida son float32."""
        data = np.random.randn(50, 3).astype(np.float32)
        X, y = BigQueryTimeSeriesAdapter._create_sequences(
            data, target_idx=0, sequence_length=5, forecast_horizon=2
        )
        assert X.dtype == np.float32
        assert y.dtype == np.float32


# ---- Tests del Pipeline Completo ----


class TestLoadData:
    """Tests del pipeline completo load_training_data / load_validation_data."""

    def _setup_mock_query(self, mock_client, dataframe):
        """Configura el mock de BigQuery para retornar el DataFrame."""
        mock_job = MagicMock()
        mock_job.to_dataframe.return_value = dataframe
        mock_client.query.return_value = mock_job

    def test_load_training_data_returns_tensors(
        self, adapter_with_mock, mock_bq_client, sample_dataframe
    ):
        """load_training_data retorna tupla de tensores PyTorch."""
        self._setup_mock_query(mock_bq_client, sample_dataframe)
        X_train, y_train = adapter_with_mock.load_training_data()
        assert isinstance(X_train, torch.Tensor)
        assert isinstance(y_train, torch.Tensor)

    def test_training_data_shapes(
        self, adapter_with_mock, mock_bq_client, sample_dataframe
    ):
        """Los tensores de entrenamiento tienen las shapes correctas."""
        self._setup_mock_query(mock_bq_client, sample_dataframe)
        X_train, y_train = adapter_with_mock.load_training_data()
        # 500 filas - (24 + 6) + 1 = 471 ventanas, 80% = 376
        assert X_train.shape[1] == 24  # sequence_length
        assert X_train.shape[2] == 7  # n_features
        assert y_train.shape[1] == 6  # forecast_horizon

    def test_validation_data_shapes(
        self, adapter_with_mock, mock_bq_client, sample_dataframe
    ):
        """Los tensores de validación tienen las shapes correctas."""
        self._setup_mock_query(mock_bq_client, sample_dataframe)
        X_val, y_val = adapter_with_mock.load_validation_data()
        assert X_val.shape[1] == 24
        assert X_val.shape[2] == 7
        assert y_val.shape[1] == 6

    def test_train_val_split_proportions(
        self, adapter_with_mock, mock_bq_client, sample_dataframe
    ):
        """Los datos se dividen correctamente según split_ratio."""
        self._setup_mock_query(mock_bq_client, sample_dataframe)
        X_train, _ = adapter_with_mock.load_training_data()
        X_val, _ = adapter_with_mock.load_validation_data()
        total = X_train.shape[0] + X_val.shape[0]
        train_ratio = X_train.shape[0] / total
        # Debe ser aprox 0.8 (exacto depende del redondeo de int)
        assert 0.75 <= train_ratio <= 0.85

    def test_data_is_cached(
        self, adapter_with_mock, mock_bq_client, sample_dataframe
    ):
        """Llamar load_training_data múltiples veces no ejecuta la query de nuevo."""
        self._setup_mock_query(mock_bq_client, sample_dataframe)
        adapter_with_mock.load_training_data()
        adapter_with_mock.load_training_data()
        adapter_with_mock.load_validation_data()
        # La query se debe haber ejecutado solo una vez
        assert mock_bq_client.query.call_count == 1

    def test_tensors_are_float32(
        self, adapter_with_mock, mock_bq_client, sample_dataframe
    ):
        """Los tensores de salida son float32."""
        self._setup_mock_query(mock_bq_client, sample_dataframe)
        X_train, y_train = adapter_with_mock.load_training_data()
        assert X_train.dtype == torch.float32
        assert y_train.dtype == torch.float32


# ---- Tests de Error Handling ----


class TestErrorHandling:
    """Tests de manejo de errores en BigQueryTimeSeriesAdapter."""

    def test_bigquery_exception_is_wrapped(
        self, adapter_with_mock, mock_bq_client
    ):
        """Una excepción de BigQuery se envuelve en RuntimeError."""
        mock_bq_client.query.side_effect = Exception("Connection timeout")
        with pytest.raises(RuntimeError, match="Error al consultar BigQuery"):
            adapter_with_mock.load_training_data()

    def test_empty_dataframe_raises_error(
        self, adapter_with_mock, mock_bq_client, feature_columns
    ):
        """Un DataFrame vacío lanza RuntimeError descriptivo."""
        empty_df = pd.DataFrame(columns=["timestamp"] + feature_columns)
        mock_job = MagicMock()
        mock_job.to_dataframe.return_value = empty_df
        mock_bq_client.query.return_value = mock_job
        with pytest.raises(RuntimeError, match="retornó 0 filas"):
            adapter_with_mock.load_training_data()
