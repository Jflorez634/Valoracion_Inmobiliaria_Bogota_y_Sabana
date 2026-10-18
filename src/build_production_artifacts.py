from __future__ import annotations
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import BallTree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

class SparseColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, mask=None):
        self.mask = mask
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.mask]

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    PARQUET_FILE, MODELS_DIR, MODEL_XGB_JSON, PREPROCESSOR_FILE, GEO_INDEX_FILE,
    DROP_LEAKAGE_COLS, DROP_MODEL_COLS, TARGET, DATE_COL, TRAIN_END
)

warnings.filterwarnings('ignore')

def safe_log(series: pd.Series) -> pd.Series:
    return np.where(series > 0, np.log(series), np.nan)

def clean_numeric_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.', regex=False).str.strip(),
                errors='coerce'
            )
    return df

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_LEAKAGE_COLS if c in df.columns], errors='ignore')
    df = clean_numeric_text_columns(df, ['Area Patio', 'Area Terraza o Balcon', 'Area Lote / Jardin'])
    if TARGET in df.columns:
        df['log_precio_deflactado'] = safe_log(df[TARGET])
    df['log_area'] = safe_log(df['Area'])
    df['area_por_alcoba'] = np.where(df['Alc.'] > 0, df['Area'] / df['Alc.'], np.nan)
    df['baños_totales'] = df['Baños Completos'] + 0.5 * df['Baños sin Terminar']
    df['tiene_patio'] = (df['Area Patio'] > 0).astype(int)
    df['tiene_terraza'] = (df['Area Terraza o Balcon'] > 0).astype(int)
    df['tiene_estudio'] = df['Estudio/Estar'].isin([
        'Estudio', 'Estar', 'Estudio/Estar ', 'Disponible/estudio', 'Espacio', 'Estar y Estudio'
    ]).astype(int)
    df['total_unidades'] = df['Prev.'] + df['Const.'] + df['Ter.']
    df['porc_terminadas'] = np.where(df['total_unidades'] > 0, df['Ter.'] / df['total_unidades'], np.nan)
    df['porc_disponibles'] = np.where(df['total_unidades'] > 0, df['D'] / df['total_unidades'], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def build_geo_index(df_raw: pd.DataFrame):
    print("Construyendo índice geoespacial sobre proyectos históricos...")
    lat_col = 'Latitud (Coordenada Real)'
    lon_col = 'Longitud\n(Coordenada Real)'
    
    commercial_cols = [
        'Oferta Lanzada M2 Proy.', '$ Of. Proyecto', 'V.Proy. Mes', 'N. Proy. Mes',
        'Cam. $', 'V.  3  Proyecto', 'Venta Trim.', 'D. Proy. Mes Ant.', 'D.Proy. Mes',
        'V. Acum. Proy.', 'V. Mes $ Inm. Tipo', 'V. Mes $ Proy.', 'V. Año $ Proy.',
        '$ Prom', '$ Of. Inm. Tipo', 'V. Total Proyecto', 'V. Total Inm. Tipo',
        'Oferta Total Inm.', 'Oferta Total Proyecto', 'V. Prom Proy.', 'Of.Ac.',
        'V. Año $ Inm. Tipo', 'Rango M2E', 'Rango Estudio Com.', 'A. Prom.', '$m2Pr.',
        'Verificacion Estado', 'Oferta Lanzada M2 Inm.', 'Ventas Año  Un.',
        'Ventas Año $ (mm)', 'Prev.', 'Const.', 'Ter.', 'D', 'N', 'V',
        'V.Acum.', 'V.  3 Meses', 'Venta Mes',
        'Area Of.  Inm. Tipo', 'Area Of.   Proyecto', 'V. Prom. Inm'
    ]
    avail_comm = [c for c in commercial_cols if c in df_raw.columns]
    
    geo_clean = df_raw.dropna(subset=[lat_col, lon_col, 'Codproyecto', 'Zona', 'Municipio']).copy()
    
    agg_dict = {
        lat_col: 'median',
        lon_col: 'median',
        'Zona': 'first',
        'Sub Zona': 'first',
        'Municipio': 'first',
        'Barrio': 'first',
        'zona_ipvn': 'first'
    }
    for c in avail_comm:
        agg_dict[c] = 'median'
        
    geo_df = geo_clean.groupby('Codproyecto').agg(agg_dict).reset_index()
    geo_df = geo_df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    
    coords_rad = np.radians(geo_df[['lat', 'lon']].values)
    tree = BallTree(coords_rad, metric='haversine')
    
    global_medians = geo_clean[avail_comm].median().to_dict()
    
    geo_payload = {
        'tree': tree,
        'geo_df': geo_df,
        'global_medians': global_medians,
        'avail_comm': avail_comm
    }
    joblib.dump(geo_payload, GEO_INDEX_FILE, compress=3)
    print(f"Índice geoespacial guardado en {GEO_INDEX_FILE} ({len(geo_df)} proyectos únicos).")
    return geo_payload

def train_and_export():
    print("Iniciando compilación de producción...")
    df_raw = pd.read_parquet(PARQUET_FILE)
    print(f"Dataset cargado: {df_raw.shape[0]} registros, {df_raw.shape[1]} columnas.")
    
    # 1. Geo Index
    build_geo_index(df_raw)
    
    # 2. Feature Engineering
    df_fe = add_feature_engineering(df_raw)
    df_fe[DATE_COL] = pd.to_datetime(df_fe[DATE_COL])
    df_fe = df_fe.drop(columns=[c for c in DROP_MODEL_COLS if c in df_fe.columns], errors='ignore')
    
    df_train = df_fe[df_fe[DATE_COL] <= TRAIN_END].dropna(subset=[TARGET]).copy()
    df_test = df_fe[df_fe[DATE_COL] > TRAIN_END].dropna(subset=[TARGET]).copy()
    
    X_train = df_train.drop(columns=[TARGET])
    y_train = df_train[TARGET].copy()
    X_test = df_test.drop(columns=[TARGET])
    y_test = df_test[TARGET].copy()
    
    print(f"Train split: {X_train.shape[0]} | Test split: {X_test.shape[0]}")
    
    # Usar el preprocesador limpio validado de los artefactos
    orig_art = joblib.load('pipeline_artifacts_20260209_030400.pkl')
    preprocessor = orig_art['preprocessor']
    mask_sel = orig_art['mask_sel'].copy()
    
    # Activar explícitamente las columnas de Tipo VIS (No Vis, Vis, Vis Renov., Vip, Vipa)
    all_fn = preprocessor.get_feature_names_out()
    vis_indices = [i for i, f in enumerate(all_fn) if 'tipo vis' in f.lower()]
    mask_sel[vis_indices] = True
    feature_names = all_fn[mask_sel]
    print(f"Features seleccionadas para producción: {len(feature_names)} (incluyendo las {len(vis_indices)} categorías de Tipo VIS)")
    
    print("Transformando matrices en formato esparso (CSR)...")
    Xt_tr = preprocessor.transform(X_train)[:, mask_sel]
    Xt_te = preprocessor.transform(X_test)[:, mask_sel]
    
    print("Entrenando XGBRegressor con hiperparámetros óptimos en matriz dispersa...")
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(Xt_tr, y_train)
    
    # Predicción sobre matriz dispersa (esencial para XGBoost con sparse input)
    preds_te = xgb.predict(Xt_te)
    r2 = r2_score(y_test, preds_te)
    mae = mean_absolute_error(y_test, preds_te)
    rmse = np.sqrt(mean_squared_error(y_test, preds_te))
    nmae = mae / y_test.mean()
    
    print("Métricas en Test Set:")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : COP {mae:,.2f}")
    print(f"  RMSE : COP {rmse:,.2f}")
    print(f"  NMAE : {nmae*100:.2f}% (Meta < 20%: {'CUMPLE' if nmae <= 0.20 else 'NO CUMPLE'})")
    
    xgb.save_model(str(MODEL_XGB_JSON))
    print(f"Modelo XGBoost guardado en formato JSON nativo: {MODEL_XGB_JSON}")
    
    prep_payload = {
        'preprocessor': preprocessor,
        'mask_sel': mask_sel,
        'feature_names': feature_names,
        'metrics_test': {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'nmae': float(nmae)
        }
    }
    joblib.dump(prep_payload, PREPROCESSOR_FILE, compress=3)
    print(f"Preprocesador guardado en: {PREPROCESSOR_FILE}")
    print("Compilación de producción completada con éxito!")

if __name__ == '__main__':
    train_and_export()
