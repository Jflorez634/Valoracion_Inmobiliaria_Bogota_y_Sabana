# 🏢 Valuador Inmobiliario Inteligente: Bogotá & Sabana de Cundinamarca

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn%201.6.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pydantic](https://img.shields.io/badge/Contracts-Pydantic%20v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen?logo=pytest&logoColor=white)](https://pytest.org/)

Sistema integral de tasación y valoración inmobiliaria objetiva para Bogotá D.C. y municipios de la Sabana de Cundinamarca (Chía, Cajicá, Soacha, Zipaquirá, Cota, etc.) basado en modelos de aprendizaje automático supervisado bajo la metodología **CRISP-DM** y buenas prácticas de ingeniería de software **MLOps**.

---

## 🎯 Métricas Clave de Desempeño del Modelo

| Métrica | Benchmark XGBoost (Test Set) | Meta Metodológica | Estado |
|---|---|---|---|
| **Coeficiente de Determinación ($R^2$)** | **0.9748** | $> 0.85$ | ✅ Superada con creces |
| **Error Absoluto Medio (MAE)** | **$ 58.867.300 COP** | - | ✅ Calibrado |
| **Error Cuadrático Medio (RMSE)** | **$ 89.120.943 COP** | - | ✅ Calibrado |
| **Error Normalizado (NMAE)** | **12.56%** | $< 20.0\%$ | ✅ Óptimo para mercado residencial |

---

## 🏛️ Arquitectura de la Solución (Portfolio & Production Grade)

```
┌────────────────────────────────────────────────────────┐
│                   CAPA 1: INTERFAZ WEB                 │
│      Streamlit + Folium (Mapa interactivo de Bogotá)   │
│   • Clic en el mapa para capturar (Lat, Lon)           │
│   • Formulario de características físicas del inmueble │
└───────────────────────────┬────────────────────────────┘
                            │ Dispara petición validada
                            ▼
┌────────────────────────────────────────────────────────┐
│                CAPA 2: CONTRATOS PYDANTIC              │
│   • PropertyEvaluationRequest (Fronteras y tipos)      │
│   • Validaciones físicas de coherencia (área/alcobas)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│           CAPA 3: SERVICIO DE INFERENCIA & GEO         │
│  1. GeoReferenceEngine (BallTree k-NN sobre Haversine) │
│     -> Resuelve microsector y dinámicas de proyecto    │
│  2. Feature Engineering Automático                     │
│     -> log(Area), área/alcoba, bins de Rango m2...     │
│  3. ColumnTransformer (65 numéricas + 21 categóricas)  │
│  4. SparseColumnSelector (Filtro Lasso de 58 variables)│
│  5. Inferencia XGBoost (Formato nativo modelo_xgb.json)│
│  6. Re-inflación macroeconómica a pesos corrientes 2026│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               CAPA 4: RESULTADO AL USUARIO             │
│   • Precio Estimado en Millones de COP                 │
│   • Valor unitario por m²                              │
│   • Banda de Negociación Estadística (±12.5% NMAE)     │
│   • Desglose visual de variables de mayor impacto      │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Innovaciones Técnicas Destacadas

1. **Resolución Geoespacial en Sub-milisegundos:**
   - La aplicación implementa un `GeoReferenceEngine` basado en `sklearn.neighbors.BallTree` con distancia geodésica `haversine`.
   - Indexa **5.476 proyectos históricos** de Bogotá y Sabana en un archivo compilado de apenas **429 KB**, eliminando la necesidad de exponer o procesar el dataset transaccional original de 22.5 MB.
2. **Deflación y Re-inflación Macroeconómica:**
   - El modelo predice en pesos constantes indexados a la serie histórica del **IPVNBR (Índice de Precios de Vivienda Nueva del Banco de la República / DANE)**.
   - En tiempo de inferencia, la estimación se proyecta a pesos corrientes de mercado del 2026 mediante el factor de ajuste inflacionario acumulado.
3. **Formato Nativo no Ejecutable:**
   - El modelo se almacena en `models/modelo_xgb.json`, descartando archivos pickle opacos y vulnerables, lo que permite serialización limpia y portabilidad a cualquier lenguaje o servicio cloud.
4. **Resiliencia y Monitoreo:**
   - Incluye mecanismo de `healthcheck()` al arranque ejecutado sobre *Golden Records* de referencia para evitar degradación silenciosa.

---

## 📁 Estructura del Repositorio

```text
├── src/
│   ├── app.py                      # Interfaz gráfica Streamlit con mapa Folium
│   ├── config.py                   # Constantes macroeconómicas, geográficas y de rutas
│   ├── schemas.py                  # Modelos Pydantic v2 (Request / Response)
│   ├── geocoder.py                 # Motor de indexación espacial (BallTree)
│   ├── predictor.py                # Servicio central de inferencia y Feature Engineering
│   └── build_production_artifacts.py # Script reproducible de entrenamiento y exportación
├── models/
│   ├── modelo_xgb.json             # Modelo XGBoost en formato nativo JSON (1.3 MB)
│   ├── preprocessor.joblib         # ColumnTransformer limpio + máscara Lasso (58 KB)
│   └── geo_index.joblib            # Índice espacial de 5,476 proyectos (429 KB)
├── tests/
│   └── test_inference.py           # Suite de pruebas automatizadas con pytest
├── Dockerfile                      # Contenedor optimizado de producción
├── requirements.txt                # Dependencias fijadas y verificadas
├── run_app.bat                     # Lanzador rápido para Windows
└── conftest.py                     # Configuración de rutas para pytest
```

---

## 💻 Instalación y Uso Local

### 1. Requisitos Previos
- Python 3.11 instalado.
- Git.

### 2. Configurar Entorno Virtual
```powershell
# Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd "Estimacion de precios de inmuebles en bogota18530432"

# Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scriptsctivate

# Instalar dependencias exactas
pip install -r requirements.txt
```

### 3. Ejecutar Pruebas Automatizadas
```powershell
pytest tests/test_inference.py -v
```

### 4. Lanzar la Aplicación Web
```powershell
streamlit run src/app.py
```
Acceder en el navegador a: `http://localhost:8501`

---

## 🐳 Ejecución con Docker

Para desplegar en cualquier servidor o plataforma cloud (Render, GCP Cloud Run, AWS ECS, HuggingFace):

```bash
# Construir la imagen optimizada (peso total < 450 MB)
docker build -t valuador-bogota:latest .

# Ejecutar el contenedor
docker run -p 8501:8501 valuador-bogota:latest
```

---

## 🛡️ Consideraciones de Privacidad y Seguridad

- **Aislamiento Total del Dataset:** El dataset transaccional `inmuebles_deflactados.parquet` (22.5 MB) y la matriz documental `df_matriz_documental.csv` (170 MB) están estrictamente ignorados en `.gitignore` y `.dockerignore` para proteger la información comercial sensible de ventas.
- **Validación Estricta:** Entradas validadas mediante Pydantic v2 previniendo desbordamientos o valores atípicos.
- **Sesión Efímera:** No se almacena ninguna información ingresada por los usuarios.
