from pydantic import BaseModel, Field, model_validator
from typing import Literal

class PropertyEvaluationRequest(BaseModel):
    latitud: float = Field(..., ge=4.469, le=5.266, description='Latitud en Bogotá o Sabana de Cundinamarca')
    longitud: float = Field(..., ge=-74.366, le=-73.792, description='Longitud en Bogotá o Sabana de Cundinamarca')
    area_construida: float = Field(..., ge=15.0, le=600.0, description='Área privada construida en m²')
    estrato: int = Field(..., ge=1, le=6, description='Estrato socioeconómico (1 al 6)')
    alcobas: int = Field(..., ge=1, le=5, description='Número de alcobas o habitaciones')
    banos_completos: int = Field(..., ge=1, le=6, description='Número de baños completos')
    banos_sin_terminar: int = Field(0, ge=0, le=2, description='Número de medios baños o sin terminar')
    garajes: int = Field(1, ge=0, le=4, description='Número de parqueaderos asignados')
    tipo_garaje: Literal['Cub. Sencillo', 'Cub. Lineal', 'Lineales', 'No tiene', 'Descubierto', 'Otro'] = 'Cub. Sencillo'
    tipo_inmueble: Literal['Apto.', 'Casa'] = 'Apto.'
    tiene_estudio: bool = Field(False, description='Cuenta con estudio, biblioteca o estar de TV')
    tiene_terraza_balcon: bool = Field(False, description='Cuenta con balcón o terraza privada')
    area_terraza_balcon: float = Field(0.0, ge=0.0, le=250.0, description='Área adicional de terraza o balcón en m²')
    tiene_patio: bool = Field(False, description='Cuenta con patio interior o jardín')
    area_patio: float = Field(0.0, ge=0.0, le=100.0, description='Área adicional de patio en m²')
    area_lote: float = Field(0.0, ge=0.0, le=800.0, description='Área de lote / jardín privado en m²')
    deposito: bool = Field(False, description='Cuenta con cuarto de depósito / bodega')
    tipo_vis: Literal['No Vis', 'Vis', 'Vis Renov.', 'Vip', 'Vip ', 'Vipa'] = Field(
        'No Vis',
        description='Clasificación de vivienda social o tradicional'
    )

    @model_validator(mode='after')
    def validate_coherence(self):
        if self.area_terraza_balcon > 0:
            self.tiene_terraza_balcon = True
        if self.area_patio > 0:
            self.tiene_patio = True
        if self.garajes == 0:
            self.tipo_garaje = 'No tiene'
        if self.tipo_vis == 'Vip':
            self.tipo_vis = 'Vip '
        return self

class PropertyEvaluationResponse(BaseModel):
    precio_corriente_cop: float = Field(..., description='Precio de mercado estimado en pesos corrientes 2026')
    precio_m2_cop: float = Field(..., description='Precio promedio estimado por metro cuadrado')
    rango_min_cop: float = Field(..., description='Límite inferior de la banda de negociación sugerida (-12.5%)')
    rango_max_cop: float = Field(..., description='Límite superior de la banda de negociación sugerida (+12.5%)')
    precio_deflactado_base: float = Field(..., description='Precio base en pesos constantes del modelo')
    municipio_detectado: str
    zona_detectada: str
    subzona_detectada: str
    barrio_detectado: str
    distancia_referencia_m: float
    fuera_de_geocerca: bool
    advertencias: list[str] = []
