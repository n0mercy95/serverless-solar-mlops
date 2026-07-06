# ============================================================
# train.py — Entrypoint para Entrenamiento en Vertex AI
# Serverless Solar MLOps | Sub-fase 2.2
# ============================================================
# Script principal orquestador del entrenamiento del Transformer
# Bi-LSTM. Instancia los puertos, adaptadores, y configuración
# del dominio para realizar un ciclo de entrenamiento resiliente
# (con checkpoints y observabilidad estructural).
# ============================================================

import os
import sys
import traceback

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Dominio
from domain.models.config import ModelConfig, TrainingConfig
from domain.models.transformer_bilstm import TransformerBiLSTM
from entrypoints.factories import LossFactory, ModelFactory

# Infraestructura (Adaptadores)
from adapters.adapter_config import BigQueryConfig, StorageConfig
from adapters.data_adapters import BigQueryTimeSeriesAdapter
from adapters.logging_adapter import CloudMetricsLoggerAdapter
from adapters.model_adapters import (
    VertexCheckpointAdapter,
    VertexModelRepositoryAdapter,
)

# Configuración del logger global para capturar todo error no manejado
logger = CloudMetricsLoggerAdapter("train_entrypoint")

def create_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Convierte los tensores en DataLoaders de PyTorch."""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Ejecuta un paso de entrenamiento sobre el dataset."""
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        
        # Squeeze si forecast_horizon == 1 y criterion lo necesita
        if predictions.shape != batch_y.shape:
            predictions = predictions.view_as(batch_y)

        loss = criterion.compute(predictions, batch_y)
        loss.backward()
        
        # Gradient clipping para estabilidad en LSTM/Transformer
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evalúa el modelo sobre el dataset de validación."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            predictions = model(batch_X)
            
            if predictions.shape != batch_y.shape:
                predictions = predictions.view_as(batch_y)
                
            loss = criterion.compute(predictions, batch_y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main() -> None:
    """Punto de entrada principal para Vertex AI."""
    
    try:
        # Carga de configuraciones desde variables de entorno si están presentes
        model_kwargs = {}
        if "MODEL_D_MODEL" in os.environ:
            model_kwargs["d_model"] = int(os.environ["MODEL_D_MODEL"])
        if "MODEL_N_HEADS" in os.environ:
            model_kwargs["n_heads"] = int(os.environ["MODEL_N_HEADS"])
        if "MODEL_N_ENCODER_LAYERS" in os.environ:
            model_kwargs["n_encoder_layers"] = int(os.environ["MODEL_N_ENCODER_LAYERS"])
        if "MODEL_LSTM_HIDDEN_SIZE" in os.environ:
            model_kwargs["lstm_hidden_size"] = int(os.environ["MODEL_LSTM_HIDDEN_SIZE"])
        if "MODEL_DROPOUT" in os.environ:
            model_kwargs["dropout"] = float(os.environ["MODEL_DROPOUT"])
        if "TRAIN_SEQUENCE_LENGTH" in os.environ:
            model_kwargs["sequence_length"] = int(os.environ["TRAIN_SEQUENCE_LENGTH"])
        if "TRAIN_FORECAST_HORIZON" in os.environ:
            model_kwargs["forecast_horizon"] = int(os.environ["TRAIN_FORECAST_HORIZON"])
        
        model_config = ModelConfig(**model_kwargs)
        
        training_kwargs = {}
        if "TRAIN_EPOCHS" in os.environ:
            training_kwargs["epochs"] = int(os.environ["TRAIN_EPOCHS"])
        if "TRAIN_BATCH_SIZE" in os.environ:
            training_kwargs["batch_size"] = int(os.environ["TRAIN_BATCH_SIZE"])
        if "TRAIN_LEARNING_RATE" in os.environ:
            training_kwargs["learning_rate"] = float(os.environ["TRAIN_LEARNING_RATE"])
            
        training_config = TrainingConfig(**training_kwargs)
        bq_config = BigQueryConfig()
        storage_config = StorageConfig()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger._logger.info(f"Iniciando entrenamiento. Dispositivo: {device}")
        
    except Exception as e:
        logger._logger.error(
            "Fallo crítico leyendo la configuración",
            exc_info=True,
            extra={"jsonPayload": {"error": str(e)}}
        )
        sys.exit(1)

    try:
        # 2. Instanciación de adaptadores
        data_adapter = BigQueryTimeSeriesAdapter(bq_config)
        model_adapter = VertexModelRepositoryAdapter(storage_config)
        checkpoint_adapter = VertexCheckpointAdapter(storage_config)

        # 3. Carga de datos
        X_train, y_train = data_adapter.load_training_data()
        X_val, y_val = data_adapter.load_validation_data()
        
        # Ajustamos `n_features` según los datos reales recibidos
        if X_train.shape[2] != model_config.n_features:
            logger._logger.info(
                f"Ajustando n_features a {X_train.shape[2]} basado en los datos"
            )
            model_config.n_features = X_train.shape[2]

        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, training_config.batch_size
        )

        # 4. Instanciación del Modelo y Estrategia (Loss) mediante Factory
        model = ModelFactory.create("transformer_bilstm", model_config).to(device)
        criterion = LossFactory.create("mae")
        optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)

        # 5. Gestión de Checkpoints (Resiliencia)
        start_epoch = 0
        best_val_loss = float("inf")
        
        latest_checkpoint = checkpoint_adapter.load_latest_checkpoint()
        if latest_checkpoint:
            logger._logger.info(
                f"Resumiendo desde el checkpoint de la época {latest_checkpoint['epoch']}"
            )
            model.load_state_dict(latest_checkpoint["model_state_dict"])
            optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])
            start_epoch = latest_checkpoint["epoch"]
            best_val_loss = latest_checkpoint.get("best_val_loss", float("inf"))

        # 6. Bucle de Entrenamiento
        for epoch in range(start_epoch + 1, training_config.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)

            # Log estructurado
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            logger.log_epoch_metrics(epoch, metrics)

            # Checkpoint en cada época
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": min(best_val_loss, val_loss),
            }
            checkpoint_adapter.save_checkpoint(checkpoint_state, epoch)

            # Lógica para "Best Model"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Metadata a guardar en model.pt/metadata.json
                final_metadata = {
                    "best_epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "model_config": model_config.model_dump(),
                    "training_config": training_config.model_dump()
                }
                
                # Salvamos el mejor candidato a Artifact Registry / GCS
                model_adapter.save_model(model, final_metadata)
                
        # 7. Finalización
        logger.log_training_complete({
            "best_val_loss": best_val_loss,
            "total_epochs": training_config.epochs
        })
        
    except Exception as e:
        logger._logger.error(
            "Fallo catastrófico en el pipeline de entrenamiento",
            exc_info=True,
            extra={
                "jsonPayload": {
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
