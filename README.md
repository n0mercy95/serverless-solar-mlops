# ☀️ Serverless Solar MLOps

**Sistema de Pronóstico Fotovoltaico Automatizado con MLOps en Google Cloud Platform**

---

## Abstract

Este proyecto implementa un sistema empresarial de **MLOps de estado del arte** para la operacionalización de un modelo **Transformer Bi-LSTM** orientado al pronóstico de generación de energía solar fotovoltaica.

El pipeline automatiza el ciclo completo de vida del modelo de machine learning:

1. **Ingesta orientada a eventos** — Los datos estructurados de la Capa Oro en BigQuery disparan automáticamente el reentrenamiento vía Cloud Pub/Sub.
2. **Entrenamiento asíncrono** — Cloud Composer (Apache Airflow) orquesta trabajos de entrenamiento en Vertex AI mediante Deferrable Operators, evitando el bloqueo de recursos.
3. **Evaluación Champion vs Challenger** — El modelo candidato se compara científicamente (MAE/RMSE) contra el modelo en producción.
4. **Despliegue condicional** — Si el candidato demuestra superioridad, se promueve automáticamente mediante Version Aliasing en Vertex AI Model Registry.

### Principios de Diseño

| Principio | Implementación |
|-----------|---------------|
| **Arquitectura Hexagonal** | Separación estricta entre dominio matemático (`domain/`) e infraestructura GCP (`adapters/`) |
| **Patrones GoF** | Factory (instanciación dinámica), Strategy (funciones de pérdida), Adapter (conexiones externas) |
| **Observabilidad** | Logging estructurado JSON obligatorio, métricas basadas en registros en Cloud Monitoring |
| **Tolerancia a fallos** | Bloques `try-except` obligatorios con Exponential Backoff y clasificación de excepciones Airflow |
| **Reproducibilidad** | Dependencias ancladas, contenedores inmutables multi-stage, entorno virtual aislado |

### Stack Tecnológico

```
PyTorch · Pandas · BigQuery · Vertex AI · Cloud Composer (Airflow)
Cloud Pub/Sub · Cloud Monitoring · Docker · Pydantic · structlog
```

---

## 📋 Estado del Proyecto

| Fase | Sub-fase | Descripción | Rama | Estado |
|------|----------|-------------|------|--------|
| **0** | **0.1** | Inicialización y Dependencias | `chore/0.1-project-setup` | ✅ Completada |
| **0** | **0.2** | Contenedorización Base | `chore/0.2-docker-setup` | ⏳ Pendiente |
| **1** | **1.1** | Core de Dominio y Puertos Hexagonales | `feat/1.1-domain-core` | ⏳ Pendiente |
| **1** | **1.2** | Implementación de Patrones GoF | `feat/1.2-gof-patterns` | ⏳ Pendiente |
| **2** | **2.1** | Adaptadores de BigQuery y Storage | `feat/2.1-io-adapters` | ⏳ Pendiente |
| **2** | **2.2** | Entrypoint y Observabilidad Estructurada | `feat/2.2-training-logging` | ⏳ Pendiente |
| **3** | **3.1** | DAG Base y Deferrable Operators | `feat/3.1-airflow-dag` | ⏳ Pendiente |
| **3** | **3.2** | Resiliencia del DAG e Idempotencia | `feat/3.2-airflow-resilience` | ⏳ Pendiente |
| **4** | **4.1** | Modelo Champion/Challenger y Aliasing | `feat/4.1-model-registry-aliasing` | ⏳ Pendiente |
| **4** | **4.2** | Log-based Metrics y Alertas Automáticas | `feat/4.2-monitoring-alerts` | ⏳ Pendiente |

---

## ✅ Avance Sub-fase 0.1 — Inicialización y Dependencias

> **Objetivo:** Establecer el entorno de desarrollo, protección de secretos y dependencias reproducibles.

### Entregables

| Archivo | Descripción |
|---------|-------------|
| `.gitignore` | Protección multicapa: secretos, credenciales GCP, datos/modelos, cache Python, IDE |
| `.env.example` | Plantilla completa de variables de entorno (GCP, Vertex AI, hiperparámetros, Pub/Sub) |
| `requirements-train.txt` | 20+ dependencias ancladas con `==` para reproducibilidad bit-a-bit |
| `venv/` | Entorno virtual Python aislado (excluido de Git) |
| `src/` | Scaffolding de Arquitectura Hexagonal |

### Estructura del Proyecto

```
serverless-solar-mlops/
├── .gitignore                  # Protección de datos y secretos
├── .env.example                # Plantilla de variables de entorno
├── README.md                   # Este archivo
├── requirements-train.txt      # Dependencias ancladas
├── docs/
│   ├── prd/
│   │   └── prd.md              # Product Requirements Document
│   └── advances/
│       └── advance-0.1.md      # Detalle completo de la sub-fase 0.1
└── src/
    ├── domain/                 # Core matemático aislado (sin deps GCP)
    │   ├── models/             # Transformer Bi-LSTM (PyTorch)
    │   ├── ports/              # Interfaces abstractas (Hexagonal)
    │   └── strategies/         # Patrón Strategy: MAE, RMSE
    ├── adapters/               # Infraestructura (BigQuery, Vertex)
    └── entrypoints/            # Punto de entrada Vertex AI
```

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd serverless-solar-mlops

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con los valores reales de tu proyecto GCP

# 4. Instalar dependencias
pip install -r requirements-train.txt
```

---

## 📖 Documentación

- [PRD Completo](docs/prd/prd.md) — Product Requirements Document con arquitectura y plan de ejecución
- [Avance Sub-fase 0.1](docs/advances/advance-0.1.md) — Detalle completo de la primera entrega

---

> *Proyecto en desarrollo activo — Última actualización: 4 de Julio, 2026*
