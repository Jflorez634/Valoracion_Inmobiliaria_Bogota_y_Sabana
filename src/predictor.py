from __future__ import annotations
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class SparseColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, mask=None):
        self.mask = mask
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.mask]

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_XGB_JSON, PREPROCESSOR_FILE, GEO_INDEX_FILE,
    IPC_REBASE_REF, FECHA_MES_REF, FACTOR_REINFLACION_CORRIENTE_2026,
    NMAE_CONFIDENCE
)
from src.schemas import PropertyEvaluationRequest, PropertyEvaluationResponse
from src.geocoder import GeoReferenceEngine

warnings.filterwarnings('ignore')

def get_rango_m2(area: float) -> str:
    if area <= 50.0:
        return '0 - 50'
    elif area <= 75.0:
        return '50 - 75'
    elif area <= 100.0:
        return '75 - 100'
    elif area <= 125.0:
        return '100 - 125'
    elif area <= 150.0:
        return '125 - 150'
    elif area <= 200.0:
        return '150 - 200'
    else:
        return ' > 200'

class PropertyValuationService:
    def __init__(
        self,
        model_path: Path | str | None = None,
        preprocessor_path: Path | str | None = None,
        geo_index_path: Path | str | None = None
    ):
        if model_path is None:
            model_path = MODEL_XGB_JSON
        if preprocessor_path is None:
            preprocessor_path = PREPROCESSOR_FILE
            
        self.geocoder = GeoReferenceEngine(geo_index_path)
        
        # Load preprocessor payload
        prep_payload = joblib.load(preprocessor_path)
        self.preprocessor = prep_payload['preprocessor']
        self.mask_sel = prep_payload['mask_sel']
        self.feature_names = prep_payload['feature_names']
        
        # Load native XGBoost model
        self.xgb = XGBRegressor()
        self.xgb.load_model(str(model_path))

    def evaluate(self, request: PropertyEvaluationRequest) -> PropertyEvaluationResponse:
        # 1. Spatial Resolution
        geo_info = self.geocoder.lookup(request.latitud, request.longitud)
        comm = geo_info['commercial_features']
        
        # 2. Assemble 86-column DataFrame matching ColumnTransformer schema
        lat_col = 'Latitud (Coordenada Real)'
        lon_col = 'Longitud\n(Coordenada Real)'
        
        # Dynamics of construction
        prev = comm.get('Prev.', 0.0)
        const = comm.get('Const.', 0.0)
        ter = comm.get('Ter.', 0.0)
        d_units = comm.get('D', 0.0)
        total_unidades = prev + const + ter
        porc_terminadas = (ter / total_unidades) if total_unidades > 0 else 0.0
        porc_disponibles = (d_units / total_unidades) if total_unidades > 0 else 0.0
        
        # Map user input to model features
        data = {
            'Estrato': request.estrato,
            'Area': request.area_construida,
            'Alc.': request.alcobas,
            'Baños Completos': request.banos_completos,
            'Baños sin Terminar': request.banos_sin_terminar,
            'Area Patio': request.area_patio,
            'Area Lote / Jardin': request.area_lote,
            'Area Terraza o Balcon': request.area_terraza_balcon,
            lat_col: request.latitud,
            lon_col: request.longitud,
            'ipc_rebase_100': IPC_REBASE_REF,
            'ipc_real_rebase_100': IPC_REBASE_REF,
            'factor': 1.0,
            
            # Engineered numeric
            'log_area': np.log(request.area_construida),
            'area_por_alcoba': request.area_construida / request.alcobas,
            'baños_totales': request.banos_completos + 0.5 * request.banos_sin_terminar,
            'tiene_patio': 1 if request.tiene_patio else 0,
            'tiene_terraza': 1 if request.tiene_terraza_balcon else 0,
            'tiene_estudio': 1 if request.tiene_estudio else 0,
            'total_unidades': total_unidades,
            'porc_terminadas': porc_terminadas,
            'porc_disponibles': porc_disponibles,
            
            # Categorical features
            'Codproyecto': geo_info['Codproyecto'],
            'Zona': geo_info['Zona'],
            'Sub Zona': geo_info['Sub Zona'],
            'Tipo': request.tipo_inmueble,
            '# Gar.': str(request.garajes),
            'Estudio/Estar': 'Estudio' if request.tiene_estudio else 'No',
            'Depósito': 'Si' if request.deposito else 'No',
            'Garaje': request.tipo_garaje,
            'Valor Garaje': 'Incluido en el precio',
            'Tipo Garaje Adicional': '0',
            'Valor Garaje Adicional': '$0',
            'Tipo VIS': request.tipo_vis,
            'Inm. Tipo Activo': 'Si',
            '$ Prom. Oferta m2': '0',
            'Rango m2': get_rango_m2(request.area_construida),
            'Estado': 'Term./',
            'fecha_mes': FECHA_MES_REF,
            'Municipio': geo_info['Municipio'],
            'Barrio': geo_info['Barrio'],
            'Certificacion Sostenible': 'No',
            'zona_ipvn': geo_info['zona_ipvn']
        }
        
        # Inject all commercial columns
        for k, v in comm.items():
            if k not in data:
                data[k] = v
                
        # Defensive fallback: ensure all expected columns exist
        for col in self.preprocessor.feature_names_in_:
            if col not in data:
                data[col] = 0.0
                
        df_row = pd.DataFrame([data])
        
        # 3. Transform via preprocessor and apply mask (returns sparse CSR)
        Xt_row = self.preprocessor.transform(df_row)[:, self.mask_sel]
        
        # 4. Predict in constant pesos
        pred_deflactado = float(self.xgb.predict(Xt_row)[0])
        pred_deflactado = max(pred_deflactado, 15_000_000.0) # floor against non-physical values
        
        # 5. Re-inflate to current market COP 2026
        precio_corriente = pred_deflactado * FACTOR_REINFLACION_CORRIENTE_2026
        precio_m2 = precio_corriente / request.area_construida
        rango_min = precio_corriente * (1.0 - NMAE_CONFIDENCE)
        rango_max = precio_corriente * (1.0 + NMAE_CONFIDENCE)
        
        advertencias = list(geo_info['advertencias'])
        if request.area_construida > 400.0:
            advertencias.append("Inmueble de gran metraje (> 400 m²). La estimación presenta mayor variabilidad.")
            
        return PropertyEvaluationResponse(
            precio_corriente_cop=round(precio_corriente, -4), # redondeo a decenas de mil
            precio_m2_cop=round(precio_m2, -2),
            rango_min_cop=round(rango_min, -4),
            rango_max_cop=round(rango_max, -4),
            precio_deflactado_base=round(pred_deflactado, 2),
            municipio_detectado=geo_info['Municipio'],
            zona_detectada=geo_info['Zona'],
            subzona_detectada=geo_info['Sub Zona'],
            barrio_detectado=geo_info['Barrio'],
            distancia_referencia_m=geo_info['distancia_referencia_m'],
            fuera_de_geocerca=geo_info['fuera_de_geocerca'],
            advertencias=advertencias
        )

    def healthcheck(self) -> bool:
        """Sanity check running on golden sample."""
        req = PropertyEvaluationRequest(
            latitud=4.6757,
            longitud=-74.0538,
            area_construida=75.0,
            estrato=5,
            alcobas=2,
            banos_completos=2,
            garajes=1,
            tipo_inmueble='Apto.'
        )
        res = self.evaluate(req)
        # Price should realistically be in 400M - 800M COP for 75m² in Chicó estrato 5
        return (350_000_000.0 <= res.precio_corriente_cop <= 1_200_000_000.0)
