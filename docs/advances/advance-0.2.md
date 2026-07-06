# 📊 Avance Sub-fase 0.2 — Contenedorización Base

> **Rama:** `chore/0.2-docker-setup`
> **Fecha:** 4 de Julio, 2026
> **Referencia:** [PRD](file:///Users/matias95lopez/Desktop/serverless-solar-mlops/docs/prd/prd.md) — Sección 5, Sub-fase 0.2

---

## Estado General del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **0.1** | Inicialización y Dependencias | ✅ Completada |
| **0.2** | Contenedorización Base | ✅ Completada |
| **1.1** | Core de Dominio y Puertos Hexagonales | ⏳ Pendiente |
| **1.2** | Implementación de Patrones GoF | ⏳ Pendiente |
| **2.1** | Adaptadores de BigQuery y Storage | ⏳ Pendiente |
| **2.2** | Entrypoint y Observabilidad Estructurada | ⏳ Pendiente |
| **3.1** | DAG Base y Deferrable Operators | ⏳ Pendiente |
| **3.2** | Resiliencia del DAG e Idempotencia | ⏳ Pendiente |
| **4.1** | Modelo Champion/Challenger y Aliasing | ⏳ Pendiente |
| **4.2** | Log-based Metrics y Alertas Automáticas | ⏳ Pendiente |

---

## Objetivo

Creación del `Dockerfile.training` (multi-stage) reduciendo la superficie de ataque, y `docker-compose.yml` simulando volúmenes de Vertex AI localmente, garantizando la reproducibilidad sin afectar la máquina host.

---

## Entregables Creados

### 1. `Dockerfile.training` — Imagen Multi-Stage Inmutable

Diseñado con una estrategia de **dos stages** para minimizar la superficie de ataque:

```
┌─────────────────────────────────────────────────┐
│  Stage 1: BUILDER                               │
│  python:3.11-slim                               │
│  ├── Instala build-essential, gcc, g++          │
│  ├── pip install --prefix=/install              │
│  └── Compila todas las dependencias             │
├─────────────────────────────────────────────────┤
│  Stage 2: RUNTIME                               │
│  python:3.11-slim (imagen limpia)               │
│  ├── COPY --from=builder /install (solo libs)   │
│  ├── Usuario no-root (mlops:1000)               │
│  ├── PYTHONPATH="/app/src"                      │
│  ├── Directorios GCS pre-creados                │
│  └── ENTRYPOINT: python -m src.entrypoints.train│
│                                                 │
│  ❌ Sin compiladores                             │
│  ❌ Sin cache de pip                             │
│  ❌ Sin usuario root                             │
└─────────────────────────────────────────────────┘
```

#### Decisiones de Diseño

| Decisión | Justificación |
|----------|---------------|
| **Multi-stage build** | La imagen runtime NO contiene `gcc`, `g++` ni `build-essential`, reduciendo ~400MB y cerrando vectores de ataque |
| **`python:3.11-slim`** | Base Debian minimalista (~50MB vs ~900MB de la imagen completa) |
| **Usuario `mlops` (UID 1000)** | Principio de mínimo privilegio — nunca ejecutar como root en producción |
| **`PYTHONPATH=/app/src`** | Permite imports hexagonales (`from domain.models import ...`) sin instalación de paquetes |
| **Variables `AIP_*` como defaults** | Vertex AI las sobreescribe automáticamente en producción; en local sirven como fallback |
| **HEALTHCHECK** | Verifica que `torch` y `pandas` están importables — detecta dependencias rotas antes de entrenar |
| **Cache de capas Docker** | `COPY requirements-train.txt` antes que `COPY src/` — reconstruye dependencias SOLO si cambian |

### 2. `docker-compose.yml` — Emulación Local de Vertex AI

Dos servicios definidos:

#### Servicio `training` (principal)

| Configuración | Valor | Propósito |
|---------------|-------|-----------|
| **env_file** | `.env` | Carga todas las variables de entorno del proyecto |
| **`AIP_*` vars** | `/gcs/*` paths | Simula la inyección automática de Vertex AI |
| **Volúmenes** | `local_volumes/*` → `/gcs/*` | Replica los mount points de GCS que Vertex AI crea |
| **`src/` montado `:ro`** | Read-only | Desarrollo iterativo sin rebuild (hot-reload del código) |
| **Memory limit** | 8GB (reserva 4GB) | Simula restricciones de recursos de Vertex AI |
| **restart: "no"** | — | El entrenamiento es finito, no debe reiniciar |
| **Logging JSON** | max 50MB, 3 archivos | Simula el formato de Cloud Logging |

#### Servicio `test` (perfil: testing)

| Configuración | Valor | Propósito |
|---------------|-------|-----------|
| **entrypoint** | `pytest` | Override del entrypoint para ejecutar tests |
| **command** | `tests/ -v --cov=src` | Tests con cobertura del código fuente |
| **profile: testing** | — | Solo se levanta con `docker compose --profile testing run test` |
| **Memory limit** | 4GB | Tests necesitan menos recursos |

#### Uso Local

```bash
# Entrenar localmente (simula Vertex AI)
docker compose up training

# Ejecutar tests en contenedor aislado
docker compose --profile testing run test

# Rebuild después de cambiar requirements
docker compose build --no-cache training
```

### 3. `.dockerignore` — Optimización del Build Context

Excluye del contexto de Docker:

| Excluido | Razón |
|----------|-------|
| `venv/`, `.venv/` | Las dependencias se instalan dentro del contenedor |
| `docs/`, `*.md` | No necesarios en la imagen de entrenamiento |
| `.env`, `.env.*` | Se inyectan como variables de entorno, no se copian |
| `local_volumes/`, `data/` | Se montan como volúmenes, no se copian |
| `tests/`, `.pytest_cache/` | No se incluyen en la imagen de producción |
| `Dockerfile.*`, `docker-compose.yml` | Evita recursión |
| `*.pt`, `*.pth`, `checkpoints/` | Artefactos de modelos — no van en la imagen |

### 4. `local_volumes/` — Directorios de Emulación GCS

```
local_volumes/
├── README.md               # Documentación del propósito
├── model-output/           # Simula AIP_MODEL_DIR
│   └── .gitkeep
├── checkpoints/            # Simula AIP_CHECKPOINT_DIR
│   └── .gitkeep
├── tensorboard-logs/       # Simula AIP_TENSORBOARD_LOG_DIR
│   └── .gitkeep
└── data/                   # Simula datos de entrada (BigQuery export)
    └── .gitkeep
```

> Los `.gitkeep` preservan la estructura en Git mientras que el `.gitignore` excluye el contenido real (datos, pesos, logs).

### 5. `.gitignore` — Actualización

Se agregó la sección `Docker Local Volumes` para excluir el contenido de `local_volumes/` pero preservar la estructura con `.gitkeep`.

---

## Estructura del Repositorio Actualizada

```
serverless-solar-mlops/
├── .dockerignore               ← [NUEVO] Optimización del build context
├── .env.example                ← (fase 0.1)
├── .gitignore                  ← [ACTUALIZADO] + local_volumes exclusion
├── Dockerfile.training         ← [NUEVO] Multi-stage inmutable
├── README.md
├── docker-compose.yml          ← [NUEVO] Emulación local de Vertex AI
├── requirements-train.txt      ← (fase 0.1)
├── docs/
│   ├── advances/
│   │   ├── advance-0.1.md      ← (fase 0.1)
│   │   └── advance-0.2.md      ← [NUEVO] Este documento
│   └── prd/
│       └── prd.md
├── local_volumes/              ← [NUEVO] Emulación de GCS volumes
│   ├── README.md
│   ├── model-output/.gitkeep
│   ├── checkpoints/.gitkeep
│   ├── tensorboard-logs/.gitkeep
│   └── data/.gitkeep
└── src/                        ← (fase 0.1 — scaffolding hexagonal)
    ├── domain/
    │   ├── models/
    │   ├── ports/
    │   └── strategies/
    ├── adapters/
    └── entrypoints/
```

---

## ⏭️ Próximo Paso: Sub-fase 1.1 — Core de Dominio y Puertos Hexagonales

**Rama:** `feat/1.1-domain-core`

Lo que se creará:
- Arquitectura del Transformer Bi-LSTM en PyTorch (`src/domain/models/`)
- Puertos abstractos de salida (interfaces) (`src/domain/ports/`)
- Tests unitarios del dominio aislado de GCP

---

> *Última actualización: 4 de Julio, 2026*
