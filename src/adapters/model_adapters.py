# ============================================================
# model_adapters.py — Adaptadores de Modelo y Checkpoints
# Serverless Solar MLOps | Sub-fase 2.1
# ============================================================
# Implementaciones concretas de ModelRepositoryPort y CheckpointPort
# (Arquitectura Hexagonal). Persisten artefactos del modelo en el
# filesystem, que en producción corresponde a los mount points de
# GCS inyectados por Vertex AI (AIP_MODEL_DIR, AIP_CHECKPOINT_DIR).
#
# IMPORTANTE (PRD §4): Todas las operaciones de I/O usan bloques
# try-except obligatorios con logging estructurado JSON.
# ============================================================

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from domain.ports.ports import CheckpointPort, ModelRepositoryPort

logger = logging.getLogger(__name__)


class VertexModelRepositoryAdapter(ModelRepositoryPort):
    """Adaptador que persiste el modelo entrenado en el filesystem.

    En producción, Vertex AI monta un bucket de GCS en ``AIP_MODEL_DIR``,
    así que este adaptador funciona idénticamente en local y en la nube
    sin código GCS-specific.

    Estructura de salida::

        {model_dir}/
        ├── model.pt          # state_dict serializado
        └── metadata.json     # métricas, config, timestamp

    Args:
        model_dir: Ruta al directorio de salida del modelo.
        model_filename: Nombre del archivo de pesos (default: model.pt).
        metadata_filename: Nombre del archivo de metadatos (default: metadata.json).
    """

    def __init__(
        self,
        model_dir: str,
        model_filename: str = "model.pt",
        metadata_filename: str = "metadata.json",
    ) -> None:
        self._model_dir = Path(model_dir)
        self._model_filename = model_filename
        self._metadata_filename = metadata_filename

    def save_model(
        self, model: nn.Module, metadata: Dict[str, object]
    ) -> str:
        """Persiste el modelo entrenado y sus metadatos.

        Args:
            model: Instancia del modelo PyTorch entrenado.
            metadata: Diccionario con métricas, config, timestamp, etc.

        Returns:
            Ruta (str) del directorio donde se guardó el modelo.

        Raises:
            RuntimeError: Si falla la serialización o escritura a disco.
        """
        try:
            self._model_dir.mkdir(parents=True, exist_ok=True)

            # Guardar state_dict del modelo
            model_path = self._model_dir / self._model_filename
            torch.save(model.state_dict(), model_path)

            # Guardar metadatos como JSON
            metadata_path = self._model_dir / self._metadata_filename
            serializable_metadata = _make_json_serializable(metadata)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)

            output_dir = str(self._model_dir)

            logger.info(
                "Modelo guardado exitosamente",
                extra={
                    "jsonPayload": {
                        "action": "model_save_complete",
                        "model_path": str(model_path),
                        "metadata_path": str(metadata_path),
                        "model_size_bytes": model_path.stat().st_size,
                    }
                },
            )

            return output_dir

        except Exception as e:
            logger.error(
                "Error guardando el modelo",
                extra={
                    "jsonPayload": {
                        "action": "model_save_error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "target_dir": str(self._model_dir),
                    }
                },
            )
            raise RuntimeError(
                f"Error al guardar modelo en {self._model_dir}: "
                f"{type(e).__name__}: {e}"
            ) from e

    def load_model(self, model_path: str) -> nn.Module:
        """Carga los pesos del modelo desde el filesystem.

        NOTA: Este método retorna el state_dict raw (como dict) en lugar
        de una instancia nn.Module completa, porque la reconstrucción del
        modelo requiere el ModelConfig y la Factory, que pertenecen al
        entrypoint. El caller debe usar::

            state_dict = adapter.load_model(path)
            model.load_state_dict(state_dict)

        Args:
            model_path: Ruta al directorio que contiene el modelo.

        Returns:
            state_dict (OrderedDict) con los pesos del modelo.

        Raises:
            RuntimeError: Si el archivo no existe o está corrupto.
        """
        try:
            weights_path = Path(model_path) / self._model_filename
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Archivo de pesos no encontrado: {weights_path}"
                )

            state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=True
            )

            logger.info(
                "Pesos del modelo cargados",
                extra={
                    "jsonPayload": {
                        "action": "model_load_complete",
                        "source_path": str(weights_path),
                        "n_tensors": len(state_dict),
                    }
                },
            )

            return state_dict

        except Exception as e:
            logger.error(
                "Error cargando el modelo",
                extra={
                    "jsonPayload": {
                        "action": "model_load_error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "source_path": model_path,
                    }
                },
            )
            raise RuntimeError(
                f"Error al cargar modelo desde {model_path}: "
                f"{type(e).__name__}: {e}"
            ) from e

    def load_metadata(self, model_path: str) -> Dict[str, object]:
        """Carga los metadatos JSON del modelo.

        Args:
            model_path: Ruta al directorio que contiene los metadatos.

        Returns:
            Diccionario con los metadatos del modelo.

        Raises:
            RuntimeError: Si el archivo no existe o el JSON es inválido.
        """
        try:
            metadata_path = Path(model_path) / self._metadata_filename
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Archivo de metadatos no encontrado: {metadata_path}"
                )

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            logger.info(
                "Metadatos del modelo cargados",
                extra={
                    "jsonPayload": {
                        "action": "metadata_load_complete",
                        "source_path": str(metadata_path),
                        "keys": list(metadata.keys()),
                    }
                },
            )

            return metadata

        except Exception as e:
            logger.error(
                "Error cargando metadatos del modelo",
                extra={
                    "jsonPayload": {
                        "action": "metadata_load_error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                },
            )
            raise RuntimeError(
                f"Error al cargar metadatos desde {model_path}: "
                f"{type(e).__name__}: {e}"
            ) from e


class VertexCheckpointAdapter(CheckpointPort):
    """Adaptador de checkpoints para tolerancia a fallos en Vertex AI.

    Persiste el estado completo del entrenamiento (pesos, optimizador,
    epoch, métricas) en el directorio ``AIP_CHECKPOINT_DIR``. Permite
    reanudar el entrenamiento si el job de Vertex AI es interrumpido.

    Convención de nombrado::

        checkpoint_epoch_001.pt
        checkpoint_epoch_002.pt
        ...

    Args:
        checkpoint_dir: Ruta al directorio de checkpoints.
    """

    _CHECKPOINT_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pt$")

    def __init__(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)

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

        Raises:
            RuntimeError: Si falla la serialización o escritura.
        """
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
            checkpoint_path = self._checkpoint_dir / filename

            torch.save(state, checkpoint_path)

            logger.info(
                "Checkpoint guardado",
                extra={
                    "jsonPayload": {
                        "action": "checkpoint_save_complete",
                        "epoch": epoch,
                        "path": str(checkpoint_path),
                        "size_bytes": checkpoint_path.stat().st_size,
                    }
                },
            )

            return str(checkpoint_path)

        except Exception as e:
            logger.error(
                "Error guardando checkpoint",
                extra={
                    "jsonPayload": {
                        "action": "checkpoint_save_error",
                        "epoch": epoch,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                },
            )
            raise RuntimeError(
                f"Error al guardar checkpoint epoch {epoch}: "
                f"{type(e).__name__}: {e}"
            ) from e

    def load_latest_checkpoint(self) -> Optional[Dict[str, object]]:
        """Carga el checkpoint más reciente disponible.

        Escanea el directorio de checkpoints, filtra los archivos
        que coinciden con el patrón ``checkpoint_epoch_NNN.pt``,
        y carga el que tiene el epoch más alto.

        Returns:
            Diccionario con el estado guardado, o None si no existe
            ningún checkpoint previo.
        """
        try:
            if not self._checkpoint_dir.exists():
                logger.info(
                    "Directorio de checkpoints no existe. Iniciando desde cero.",
                    extra={
                        "jsonPayload": {
                            "action": "checkpoint_dir_not_found",
                            "dir": str(self._checkpoint_dir),
                        }
                    },
                )
                return None

            # Buscar todos los checkpoints válidos
            checkpoints = []
            for f in self._checkpoint_dir.iterdir():
                match = self._CHECKPOINT_PATTERN.match(f.name)
                if match:
                    epoch_num = int(match.group(1))
                    checkpoints.append((epoch_num, f))

            if not checkpoints:
                logger.info(
                    "No se encontraron checkpoints previos",
                    extra={
                        "jsonPayload": {
                            "action": "no_checkpoints_found",
                            "dir": str(self._checkpoint_dir),
                        }
                    },
                )
                return None

            # Ordenar por epoch (mayor primero) y tomar el más reciente
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            latest_epoch, latest_path = checkpoints[0]

            state = torch.load(
                latest_path, map_location="cpu", weights_only=False
            )

            logger.info(
                "Checkpoint cargado exitosamente",
                extra={
                    "jsonPayload": {
                        "action": "checkpoint_load_complete",
                        "epoch": latest_epoch,
                        "path": str(latest_path),
                        "total_checkpoints_found": len(checkpoints),
                    }
                },
            )

            return state

        except Exception as e:
            logger.error(
                "Error cargando checkpoint",
                extra={
                    "jsonPayload": {
                        "action": "checkpoint_load_error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                },
            )
            raise RuntimeError(
                f"Error al cargar checkpoint: {type(e).__name__}: {e}"
            ) from e


def _make_json_serializable(obj: object) -> object:
    """Convierte recursivamente un objeto a un tipo JSON-serializable.

    Maneja tipos comunes de PyTorch/numpy que json.dump no soporta:
    - torch.Tensor → float/list
    - numpy scalars → float/int
    - numpy arrays → list

    Args:
        obj: Objeto a convertir.

    Returns:
        Versión JSON-serializable del objeto.
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.tolist()
    # numpy types
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj
