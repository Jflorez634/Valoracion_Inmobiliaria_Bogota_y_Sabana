from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
PARQUET_FILE = BASE_DIR / 'inmuebles_deflactados.parquet'

MODEL_XGB_JSON = MODELS_DIR / 'modelo_xgb.json'
PREPROCESSOR_FILE = MODELS_DIR / 'preprocessor.joblib'
GEO_INDEX_FILE = MODELS_DIR / 'geo_index.joblib'

# Macroeconomic & Temporal Calibration
IPC_REBASE_REF = 102.512825
FECHA_MES_REF = '2025-04-01'
FACTOR_REINFLACION_CORRIENTE_2026 = 1.085
NMAE_CONFIDENCE = 0.1257

# Geographic Bounds (Bogota & Sabana de Cundinamarca)
LAT_MIN = 4.4690
LAT_MAX = 5.2660
LON_MIN = -74.3660
LON_MAX = -73.7920
BOGOTA_CENTER_LAT = 4.6500
BOGOTA_CENTER_LON = -74.0800
MAX_NEAREST_DISTANCE_KM = 8.0

# Feature Engineering & Pipeline Constants
TARGET = 'precio_deflactado'
DATE_COL = 'fecha_mes'
TRAIN_END = '2024-12-01'

DROP_LEAKAGE_COLS = [
    '', 'Rango $', 'Rango ', 'Rango ', 'Rango ',
    'Valor SMLV', 'Observación valor SMLV'
]

DROP_MODEL_COLS = [
    'Codinmueble', 'Proyecto', 'Vende', 'Codvende', 'P', 'log_precio_deflactado'
]
