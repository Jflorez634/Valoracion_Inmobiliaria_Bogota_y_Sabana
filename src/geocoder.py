from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    GEO_INDEX_FILE, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    MAX_NEAREST_DISTANCE_KM
)

EARTH_RADIUS_M = 6371000.0

class GeoReferenceEngine:
    def __init__(self, geo_index_path: Path | str | None = None):
        if geo_index_path is None:
            geo_index_path = GEO_INDEX_FILE
        payload = joblib.load(geo_index_path)
        self.tree = payload['tree']
        self.geo_df = payload['geo_df']
        self.global_medians = payload['global_medians']
        self.avail_comm = payload['avail_comm']
        
    def is_in_bounding_box(self, lat: float, lon: float) -> bool:
        return (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX)

    def lookup(self, lat: float, lon: float) -> dict:
        in_bbox = self.is_in_bounding_box(lat, lon)
        
        # Query BallTree with Haversine distance (expects coordinates in radians)
        coords_rad = np.radians([[lat, lon]])
        dist_rad, ind = self.tree.query(coords_rad, k=1)
        
        dist_m = float(dist_rad[0][0]) * EARTH_RADIUS_M
        nearest_row = self.geo_df.iloc[ind[0][0]]
        
        is_covered = in_bbox and (dist_m <= (MAX_NEAREST_DISTANCE_KM * 1000.0))
        
        result = {
            'Codproyecto': str(nearest_row['Codproyecto']),
            'Municipio': str(nearest_row['Municipio']),
            'Zona': str(nearest_row['Zona']),
            'Sub Zona': str(nearest_row['Sub Zona']),
            'Barrio': str(nearest_row['Barrio']),
            'zona_ipvn': str(nearest_row['zona_ipvn']),
            'lat_referencia': float(nearest_row['lat']),
            'lon_referencia': float(nearest_row['lon']),
            'distancia_referencia_m': round(dist_m, 1),
            'fuera_de_geocerca': not is_covered,
            'advertencias': []
        }
        
        if not is_covered:
            result['advertencias'].append(
                f"Ubicación a {dist_m/1000.0:.1f} km del proyecto modelado más cercano. "
                "Se aplican parámetros de imputación global de referencia."
            )
            commercial_values = self.global_medians.copy()
        else:
            commercial_values = {c: float(nearest_row[c]) for c in self.avail_comm}
            
        result['commercial_features'] = commercial_values
        return result
