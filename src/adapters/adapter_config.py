# ============================================================
# adapter_config.py — Configuración de Adaptadores (Pydantic)
# Serverless Solar MLOps | Sub-fase 2.1
# ============================================================
# Esquemas de validación para los adaptadores de infraestructura.
# Lee variables de entorno inyectadas por Vertex AI o definidas
# en el archivo .env para desarrollo local.
# ============================================================

import os
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BigQueryConfig(BaseModel):
    """Configuración para el adaptador de BigQuery (Capa Oro).

    Encapsula los parámetros de conexión y consulta a la tabla
    de series de tiempo fotovoltaicas en BigQuery.

    Attributes:
        project_id: ID del proyecto GCP.
        dataset: Nombre del dataset en BigQuery (Capa Oro).
        table: Nombre de la tabla de series de tiempo.
        feature_columns: Lista de columnas de features a extraer.
        target_column: Columna objetivo (variable a predecir).
        timestamp_column: Columna de marca temporal para ordenamiento.
        split_ratio: Proporción de datos para entrenamiento (0.0-1.0).
    """

    project_id: str = Field(
        default_factory=lambda: os.environ.get("GCP_PROJECT_ID", ""),
        description="ID del proyecto GCP",
    )
    dataset: str = Field(
        default_factory=lambda: os.environ.get("BQ_DATASET_GOLD", "gold_layer"),
        description="Nombre del dataset en BigQuery",
    )
    table: str = Field(
        default_factory=lambda: os.environ.get(
            "BQ_TABLE_TIMESERIES", "solar_timeseries"
        ),
        description="Nombre de la tabla de series de tiempo",
    )
    feature_columns: List[str] = Field(
        default=[
            "ghi",
            "dni",
            "dhi",
            "temperature",
            "humidity",
            "wind_speed",
            "power_output",
        ],
        description=(
            "Columnas de features a extraer. Incluye la variable "
            "target porque se usa como feature en la ventana de entrada."
        ),
    )
    target_column: str = Field(
        default="power_output",
        description="Columna objetivo a predecir",
    )
    timestamp_column: str = Field(
        default="timestamp",
        description="Columna de timestamp para ordenamiento cronológico",
    )
    split_ratio: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Proporción de datos para entrenamiento (resto para validación)",
    )

    @field_validator("project_id")
    @classmethod
    def project_id_not_empty(cls, v: str) -> str:
        """Valida que el project_id no esté vacío en tiempo de uso."""
        if not v or not v.strip():
            raise ValueError(
                "GCP_PROJECT_ID no está configurado. "
                "Defínelo en .env o como variable de entorno."
            )
        return v.strip()

    @field_validator("target_column")
    @classmethod
    def target_must_be_in_features(cls, v: str, info) -> str:
        """Valida que la columna target esté incluida en feature_columns."""
        features = info.data.get("feature_columns", [])
        if features and v not in features:
            raise ValueError(
                f"target_column '{v}' debe estar incluida en "
                f"feature_columns: {features}"
            )
        return v


class StorageConfig(BaseModel):
    """Configuración para los adaptadores de almacenamiento (GCS/local).

    Lee las rutas de los directorios de Vertex AI que son inyectadas
    como variables de entorno en producción, o montadas como volúmenes
    locales en desarrollo (vía docker-compose).

    Attributes:
        model_dir: Ruta al directorio de salida del modelo.
        checkpoint_dir: Ruta al directorio de checkpoints.
        model_filename: Nombre del archivo de pesos del modelo.
        metadata_filename: Nombre del archivo de metadatos JSON.
    """

    model_dir: str = Field(
        default_factory=lambda: os.environ.get(
            "AIP_MODEL_DIR", "/gcs/model-output"
        ),
        description="Directorio de salida del modelo (AIP_MODEL_DIR en Vertex AI)",
    )
    checkpoint_dir: str = Field(
        default_factory=lambda: os.environ.get(
            "AIP_CHECKPOINT_DIR", "/gcs/checkpoints"
        ),
        description="Directorio de checkpoints (AIP_CHECKPOINT_DIR en Vertex AI)",
    )
    model_filename: str = Field(
        default="model.pt",
        description="Nombre del archivo de pesos del modelo",
    )
    metadata_filename: str = Field(
        default="metadata.json",
        description="Nombre del archivo de metadatos",
    )

    @field_validator("model_dir", "checkpoint_dir")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        """Valida que las rutas no estén vacías."""
        if not v or not v.strip():
            raise ValueError(
                "La ruta del directorio no puede estar vacía. "
                "Verifica las variables AIP_MODEL_DIR / AIP_CHECKPOINT_DIR."
            )
        return v.strip()
