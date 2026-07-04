Product Requirements Document (PRD): Sistema de Pronóstico Fotovoltaico Automatizado (MLOps)
Autor: Staff MLOps Engineer & Tech Lead Estado: Aprobado para Ejecución Fecha: 4 de Julio, 2026

1. Resumen Ejecutivo y Conexión Científica
La operacionalización de modelos predictivos avanzados basados en arquitecturas profundas, como nuestro Transformer Bi-LSTM para pronóstico fotovoltaico, presenta una complejidad arquitectónica alta debido a la sensibilidad secuencial de los datos y al riesgo de degradación del modelo por concept drift.
Este PRD define el diseño de un sistema empresarial MLOps de "estado del arte" en Google Cloud Platform (GCP) para asegurar el entrenamiento continuo (Continuous Training - CT). El pipeline orquestará, mediante Apache Airflow (Cloud Composer), la extracción de los flujos de datos limpios y enriquecidos desde nuestra Capa Oro en BigQuery, y despachará el trabajo pesado de entrenamiento a Vertex AI. Al finalizar la convergencia de tensores, se implementará una evaluación científica del modelo candidato frente al modelo actualmente en producción, utilizando como referencia las métricas de precisión técnica (MAE/RMSE). Si el modelo candidato resulta superior, este será promovido y registrado automáticamente en el Vertex AI Model Registry.

2. Arquitectura del Sistema MLOps (BigQuery -> Cloud Composer -> Vertex AI)
Para aislar el plano de control del plano de cómputo, nuestra arquitectura se rige por un esquema de orquestación donde Airflow actúa exclusivamente como despachador de tareas.
Ingesta de Datos Orientada a Eventos: La canalización se iniciará cuando lleguen nuevos lotes de datos estructurados a la Capa Oro de BigQuery, lo cual publicará un evento en Cloud Pub/Sub y disparará el DAG de Airflow a través de una Cloud Function ligera.
Entrenamiento Asíncrono en Vertex AI: Debido a que el entrenamiento del Transformer Bi-LSTM toma varias horas, Airflow invocará a Vertex AI utilizando Operadores Diferibles (Deferrable Operators) (ej., CreateCustomContainerTrainingJobOperator(deferrable=True)). Esto evita el bloqueo de los worker slots del clúster orquestador delegando el sondeo a un componente asíncrono (Triggerer) y minimizando el costo operativo.
Registro y Despliegue Condicional: Finalizado el entrenamiento, el DAG ejecuta una evaluación cruzada (Champion vs Challenger). Mediante el uso de Alias de Versión (Version Aliasing), el sistema etiqueta al nuevo Transformer Bi-LSTM como candidate. Si este modelo mejora el MAE/RMSE del modelo stable, Airflow orquesta el intercambio de alias y actualiza el Endpoint de Vertex AI.

3. Diseño de Software: Arquitectura Hexagonal y Patrones GoF
La base de código será empaquetada siguiendo la Arquitectura Hexagonal (Puertos y Adaptadores) para evitar que la lógica matemática se acople a las librerías de GCP.
Patrones GoF (Gang of Four) en ML
El código dentro de nuestro contenedor de entrenamiento hará uso intensivo de patrones orientados a objetos:
Factory (Creacional): Encapsulará la instanciación dinámica del modelo (Transformer vs LSTM base) combinando fábricas con validadores estructurales como Pydantic.
Strategy (Comportamental): Abstraerá las funciones de pérdida matemática (ej. MAELossStrategy o RMSELossStrategy), permitiendo alternar la optimización sin alterar el ciclo de entrenamiento.
Adapter (Estructural): Estandarizará las conexiones externas. Tendremos un BigQueryTimeSeriesAdapter para transformar la salida SQL a un DataFrame/Tensor universal, logrando cumplir el Principio de Abierto/Cerrado.
Estructura del Repositorio (Árbol ASCII)
.
├── docker-compose.yml              # Emulación local de variables e inyecciones de Vertex AI
├── Dockerfile.training             # Imagen base inmutable (minimalista, multi-stage)
├── requirements-train.txt          # Dependencias ancladas (torch, pandas, google-cloud)
└── src/
    ├── domain/                     # CORE MATEMÁTICO AISLADO (Ninguna dep. de GCP)
    │   ├── models/                 # Arquitectura Transformer Bi-LSTM (PyTorch)
    │   ├── ports/                  # Interfaces Inbound/Outbound abstractas
    │   └── strategies/             # Patrón Strategy: MAE, RMSE Loss Functions
    ├── adapters/                   # INFRAESTRUCTURA (BigQuery, Vertex Storage)
    │   ├── data_adapters.py        # Patrón Adapter: BigQueryTimeSeriesAdapter
    │   └── model_adapters.py       # Patrón Adapter: ArtifactRegistryPort Implementation
    └── entrypoints/
        ├── factories.py            # Patrón Factory: Construcción dinámica del modelo
        └── train.py                # Punto de entrada para Vertex AI


4. Estrategia de Reentrenamiento y Observabilidad
El sistema debe tener un diseño de tolerancia a fallos de grado corporativo. Para ello, se prohíben las llamadas de infraestructura sin bloques try-except obligatorios tanto en el código de Vertex AI como en el orquestador Airflow.
Resiliencia en Airflow: Se utilizará AirflowException para reintentos mediante Exponential Backoff ante fallas de red, y AirflowFailException para abortar de inmediato ante errores estructurales irreversibles. Exigimos AirflowSkipException si las reglas de negocio determinan que la degradación del modelo en la Capa Oro aún no justifica el consumo de GPUs para un reentrenamiento.
Logging Estructurado JSON: Se prohíben los print() convencionales. Exigimos el uso de la clase customizada CloudStructuredLogFormatter que extiende la biblioteca logging de Python. Todos los logs dentro de los bloques try-except generarán un payload JSON. El agente de la nube tomará métricas del jsonPayload, identificará la gravedad (severity) y consolidará la traza.
Métricas basadas en registros: A través de este logging estructurado, enviaremos la pérdida por cada época y la métrica de validación (MAE/RMSE). Esto nos permite diseñar Log-based Metrics en Cloud Monitoring para detectar tempranamente si el modelo diverge e interrumpir proactivamente el trabajo de Vertex AI.

5. Plan de Ejecución basado en Git (Milestones y Sub-fases)
Para mitigar riesgos y asegurar la revisión de código incremental, el proyecto se construirá bajo la siguiente planificación estricta en ramas del repositorio central:
Fase 0: Configuración Base y Entorno de Desarrollo (Setup)
Sub-fase 0.1: Inicialización y Dependencias | Rama: chore/0.1-project-setup | Por qué: Establecer el entorno virtual (venv), archivo .env base para secretos, configuración estricta de .gitignore para proteger datos y la lista de dependencias ancladas (requirements-train.txt).
Sub-fase 0.2: Contenedorización Base | Rama: chore/0.2-docker-setup | Por qué: Creación del Dockerfile (multi-stage) reduciendo la superficie de ataque, y docker-compose.yml simulando volúmenes de Vertex AI localmente, garantizando la reproducibilidad sin afectar la máquina host.
Fase 1: Empaquetado del Modelo Transformer Bi-LSTM
Sub-fase 1.1: Core de Dominio y Puertos Hexagonales | Rama: feat/1.1-domain-core | Por qué: Aislar puramente la lógica algorítmica del Transformer Bi-LSTM (en PyTorch) bajo el directorio domain/, creando los puertos abstractos de salida (interfaces) para garantizar alta testabilidad desacoplada de la nube.
Sub-fase 1.2: Implementación de Patrones GoF | Rama: feat/1.2-gof-patterns | Por qué: Programar las familias de evaluación mediante el patrón Strategy (MAE/RMSE) y la lógica de instanciación con el patrón Factory para flexibilizar la experimentación de hiperparámetros del modelo matemático.
Fase 2: Configuración de Vertex AI Custom Training y Model Registry
Sub-fase 2.1: Adaptadores de BigQuery y Storage | Rama: feat/2.1-io-adapters | Por qué: Implementar los Adaptadores estructurales concretos que cumplen los contratos del dominio, integrando el SDK de GCP para extraer series de tiempo de BigQuery y exportar pesos a Artifact Registry.
Sub-fase 2.2: Entrypoint y Observabilidad Estructurada | Rama: feat/2.2-training-logging | Por qué: Codificar train.py consolidando los bloques try-except obligatorios y configurando el CloudStructuredLogFormatter para emitir eventos en formato JSON compatibles con Google Cloud Logging.
Fase 3: Orquestación del DAG en Cloud Composer / Airflow
Sub-fase 3.1: DAG Base y Deferrable Operators | Rama: feat/3.1-airflow-dag | Por qué: Ensamblar la tubería automatizada utilizando operadores diferibles hacia Vertex AI, asegurando que los workers de Airflow no se bloqueen asíncronamente mientras el Transformer entrena.
Sub-fase 3.2: Resiliencia del DAG e Idempotencia | Rama: feat/3.2-airflow-resilience | Por qué: Integrar reglas de control de flujo en Airflow (AirflowFailException, AirflowSkipException) e integrar notificaciones vía Slack de acuerdo al estado del Pipeline, dotando al orquestador de tolerancia a fallos transitorios.
Fase 4: Despliegue de Endpoints y Monitorización
Sub-fase 4.1: Modelo Champion/Challenger y Aliasing | Rama: feat/4.1-model-registry-aliasing | Por qué: Orquestar el paso analítico en Airflow que compara el MAE/RMSE y actualiza el Version Aliasing a stable, permitiendo el despliegue automático hacia los endpoints si se demuestra superioridad del candidato.
Sub-fase 4.2: Log-based Metrics y Alertas Automáticas | Rama: feat/4.2-monitoring-alerts | Por qué: Activar las métricas en Cloud Monitoring que analicen el jsonPayload extraído durante la inferencia y entrenamiento para disparar alertas a los ingenieros MLOps en caso de una divergencia algorítmica temprana.

