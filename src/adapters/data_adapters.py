# ============================================================
# data_adapters.py — Adaptador de Datos BigQuery (GoF Adapter)
# Serverless Solar MLOps | Sub-fase 2.1
# ============================================================
# Implementación concreta del DataPort (Arquitectura Hexagonal).
# Extrae series de tiempo fotovoltaicas de la Capa Oro en BigQuery,
# las transforma en tensores PyTorch con ventanas deslizantes, y
# las particiona en conjuntos de entrenamiento y validación.
#
# IMPORTANTE (PRD §4): Todas las llamadas de infraestructura usan
# bloques try-except obligatorios con logging estructurado JSON.
# ============================================================

import logging
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from google.cloud import bigquery

from domain.ports.ports import DataPort

logger = logging.getLogger(__name__)


class BigQueryTimeSeriesAdapter(DataPort):
    """Adaptador que conecta BigQuery (Capa Oro) con el dominio.

    Cumple el contrato de ``DataPort`` transformando las filas SQL
    de la tabla ``solar_timeseries`` a tensores PyTorch listos para
    el entrenamiento del Transformer Bi-LSTM.

    Pipeline interno:
        1. Query SQL parametrizada → DataFrame
        2. DataFrame → numpy array ordenado por timestamp
        3. Windowing de ventanas deslizantes (sliding windows)
        4. Split train/val según ``split_ratio``
        5. Conversión a ``torch.Tensor`` float32

    Args:
        project_id: ID del proyecto GCP.
        dataset: Nombre del dataset en BigQuery.
        table: Nombre de la tabla de series de tiempo.
        feature_columns: Lista de columnas de features a extraer.
        target_column: Columna objetivo (variable a predecir).
        timestamp_column: Columna de timestamp para ordenamiento.
        sequence_length: Longitud de la ventana de entrada.
        forecast_horizon: Número de pasos a predecir.
        split_ratio: Proporción para entrenamiento (0.0 - 1.0).
        client: Cliente de BigQuery (inyectable para testing).
    """

    def __init__(
        self,
        project_id: str,
        dataset: str,
        table: str,
        feature_columns: List[str],
        target_column: str,
        timestamp_column: str,
        sequence_length: int,
        forecast_horizon: int,
        split_ratio: float = 0.8,
        client: bigquery.Client | None = None,
    ) -> None:
        self._project_id = project_id
        self._dataset = dataset
        self._table = table
        self._feature_columns = feature_columns
        self._target_column = target_column
        self._timestamp_column = timestamp_column
        self._sequence_length = sequence_length
        self._forecast_horizon = forecast_horizon
        self._split_ratio = split_ratio

        # Inyección de dependencia: permite pasar un mock en tests
        self._client = client

        # Cache interna: se carga una sola vez y se particiona
        self._X_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None
        self._X_val: torch.Tensor | None = None
        self._y_val: torch.Tensor | None = None

    def _get_client(self) -> bigquery.Client:
        """Obtiene o crea el cliente de BigQuery (lazy initialization).

        Returns:
            Instancia de bigquery.Client.
        """
        if self._client is None:
            self._client = bigquery.Client(project=self._project_id)
        return self._client

    def _build_query(self) -> str:
        """Construye la query SQL parametrizada para la Capa Oro.

        Extrae todas las columnas de features más el timestamp,
        ordenando cronológicamente para preservar la secuencialidad
        temporal necesaria para el windowing.

        Returns:
            String SQL formateado.
        """
        columns = [self._timestamp_column] + self._feature_columns
        columns_sql = ", ".join(columns)
        return (
            f"SELECT {columns_sql} "
            f"FROM `{self._project_id}.{self._dataset}.{self._table}` "
            f"ORDER BY {self._timestamp_column} ASC"
        )

    def _query_bigquery(self) -> pd.DataFrame:
        """Ejecuta la query contra BigQuery y retorna un DataFrame.

        Returns:
            DataFrame con las columnas de features ordenadas por timestamp.

        Raises:
            RuntimeError: Si la query falla o retorna datos vacíos.
        """
        client = self._get_client()
        query = self._build_query()

        logger.info(
            "Ejecutando query contra BigQuery",
            extra={
                "jsonPayload": {
                    "action": "bigquery_query_start",
                    "project": self._project_id,
                    "dataset": self._dataset,
                    "table": self._table,
                    "n_features": len(self._feature_columns),
                }
            },
        )

        start_time = time.monotonic()

        try:
            query_job = client.query(query)
            df = query_job.to_dataframe()
        except Exception as e:
            logger.error(
                "Error ejecutando query de BigQuery",
                extra={
                    "jsonPayload": {
                        "action": "bigquery_query_error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                },
            )
            raise RuntimeError(
                f"Error al consultar BigQuery: {type(e).__name__}: {e}"
            ) from e

        elapsed = time.monotonic() - start_time

        if df.empty:
            raise RuntimeError(
                f"La tabla {self._dataset}.{self._table} retornó 0 filas. "
                "Verifica que la Capa Oro contenga datos."
            )

        logger.info(
            "Query de BigQuery completada",
            extra={
                "jsonPayload": {
                    "action": "bigquery_query_complete",
                    "rows_fetched": len(df),
                    "elapsed_seconds": round(elapsed, 3),
                    "columns": list(df.columns),
                }
            },
        )

        return df

    @staticmethod
    def _create_sequences(
        data: np.ndarray,
        target_idx: int,
        sequence_length: int,
        forecast_horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Genera ventanas deslizantes (sliding windows) sobre la serie temporal.

        Para cada posición válida ``i``, crea:
        - X: ``data[i : i + sequence_length]`` (todas las features)
        - y: ``data[i + sequence_length : i + sequence_length + forecast_horizon, target_idx]``

        Args:
            data: Array numpy shape (N, n_features).
            target_idx: Índice de la columna target dentro del array.
            sequence_length: Longitud de la ventana de entrada.
            forecast_horizon: Pasos futuros a predecir.

        Returns:
            Tupla (X, y) donde:
            - X: shape (n_windows, sequence_length, n_features)
            - y: shape (n_windows, forecast_horizon)

        Raises:
            ValueError: Si no hay suficientes datos para crear al menos
                        una ventana completa.
        """
        total_window = sequence_length + forecast_horizon
        n_samples = len(data)

        if n_samples < total_window:
            raise ValueError(
                f"Datos insuficientes para crear ventanas: "
                f"{n_samples} filas disponibles, pero se necesitan al menos "
                f"{total_window} (sequence_length={sequence_length} + "
                f"forecast_horizon={forecast_horizon})."
            )

        n_windows = n_samples - total_window + 1
        X = np.zeros((n_windows, sequence_length, data.shape[1]), dtype=np.float32)
        y = np.zeros((n_windows, forecast_horizon), dtype=np.float32)

        for i in range(n_windows):
            X[i] = data[i : i + sequence_length]
            y[i] = data[
                i + sequence_length : i + sequence_length + forecast_horizon,
                target_idx,
            ]

        return X, y

    def _prepare_data(self) -> None:
        """Pipeline completo: query → windowing → split → tensors.

        Carga los datos de BigQuery, los transforma en ventanas
        deslizantes, y los particiona en train/val. El resultado
        se almacena en cache interna para evitar múltiples queries.
        """
        if self._X_train is not None:
            return  # Ya cargado

        # 1. Query BigQuery
        df = self._query_bigquery()

        # 2. Extraer features como numpy array (ordenado por timestamp)
        feature_data = df[self._feature_columns].values.astype(np.float32)
        target_idx = self._feature_columns.index(self._target_column)

        # 3. Crear ventanas deslizantes
        X, y = self._create_sequences(
            data=feature_data,
            target_idx=target_idx,
            sequence_length=self._sequence_length,
            forecast_horizon=self._forecast_horizon,
        )

        # 4. Split train/val
        split_idx = int(len(X) * self._split_ratio)

        if split_idx == 0 or split_idx >= len(X):
            raise ValueError(
                f"Split inválido: {split_idx} de {len(X)} ventanas. "
                f"Ajusta split_ratio ({self._split_ratio}) o agrega más datos."
            )

        # 5. Convertir a tensores PyTorch
        self._X_train = torch.from_numpy(X[:split_idx])
        self._y_train = torch.from_numpy(y[:split_idx])
        self._X_val = torch.from_numpy(X[split_idx:])
        self._y_val = torch.from_numpy(y[split_idx:])

        logger.info(
            "Datos preparados para entrenamiento",
            extra={
                "jsonPayload": {
                    "action": "data_preparation_complete",
                    "total_windows": len(X),
                    "train_samples": split_idx,
                    "val_samples": len(X) - split_idx,
                    "X_train_shape": list(self._X_train.shape),
                    "y_train_shape": list(self._y_train.shape),
                    "X_val_shape": list(self._X_val.shape),
                    "y_val_shape": list(self._y_val.shape),
                }
            },
        )

    def load_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carga los datos de entrenamiento.

        Returns:
            Tupla (X_train, y_train) donde:
            - X_train: Tensor shape (n_samples, sequence_length, n_features)
            - y_train: Tensor shape (n_samples, forecast_horizon)
        """
        self._prepare_data()
        assert self._X_train is not None and self._y_train is not None
        return self._X_train, self._y_train

    def load_validation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carga los datos de validación.

        Returns:
            Tupla (X_val, y_val) con las mismas shapes que load_training_data.
        """
        self._prepare_data()
        assert self._X_val is not None and self._y_val is not None
        return self._X_val, self._y_val
