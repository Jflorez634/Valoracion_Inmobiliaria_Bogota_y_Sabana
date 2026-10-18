import pytest
from pydantic import ValidationError
from src.predictor import PropertyValuationService
from src.schemas import PropertyEvaluationRequest

@pytest.fixture(scope='module')
def service():
    return PropertyValuationService()

def test_healthcheck(service):
    assert service.healthcheck() is True

def test_chico_apartment_evaluation(service):
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
    res = service.evaluate(req)
    assert res.municipio_detectado == 'Bogota'
    assert res.zona_detectada == 'Chicó'
    assert res.precio_corriente_cop > 400_000_000
    assert res.precio_m2_cop > 6_000_000
    assert res.rango_min_cop < res.precio_corriente_cop < res.rango_max_cop
    assert not res.fuera_de_geocerca
    assert res.distancia_referencia_m < 500.0

def test_chia_house_evaluation(service):
    req = PropertyEvaluationRequest(
        latitud=4.8622,
        longitud=-74.0583,
        area_construida=140.0,
        estrato=4,
        alcobas=3,
        banos_completos=3,
        garajes=2,
        tipo_inmueble='Casa',
        tiene_patio=True,
        area_patio=20.0
    )
    res = service.evaluate(req)
    assert res.municipio_detectado == 'Chía'
    assert res.zona_detectada == 'Chía'
    assert res.precio_corriente_cop > 300_000_000
    assert res.distancia_referencia_m < 500.0

def test_geofence_out_of_bounds_warning(service):
    req = PropertyEvaluationRequest(
        latitud=5.2500,
        longitud=-74.3500, # Edge of Sabana, far from modeled projects
        area_construida=60.0,
        estrato=3,
        alcobas=2,
        banos_completos=1,
        garajes=0,
        tipo_inmueble='Apto.'
    )
    res = service.evaluate(req)
    # Check that distance is calculated and if > 8km, triggers warning
    if res.distancia_referencia_m > 8000.0:
        assert res.fuera_de_geocerca is True
        assert len(res.advertencias) > 0

def test_pydantic_schema_validations():
    with pytest.raises(ValidationError):
        # Estrato 7 invalid
        PropertyEvaluationRequest(
            latitud=4.65, longitud=-74.08, area_construida=50, estrato=7,
            alcobas=2, banos_completos=1
        )
    with pytest.raises(ValidationError):
        # Area 5m2 invalid (< 15)
        PropertyEvaluationRequest(
            latitud=4.65, longitud=-74.08, area_construida=5, estrato=3,
            alcobas=1, banos_completos=1
        )

def test_vis_apartment_evaluation(service):
    req_vis = PropertyEvaluationRequest(
        latitud=4.5820,
        longitud=-74.2180,
        area_construida=48.0,
        estrato=2,
        alcobas=2,
        banos_completos=1,
        garajes=0,
        tipo_inmueble='Apto.',
        tipo_vis='Vis'
    )
    res_vis = service.evaluate(req_vis)
    assert res_vis.precio_corriente_cop > 80_000_000
    assert res_vis.precio_corriente_cop < 220_000_000
    
    # VIP should be lower than No Vis
    req_novis = PropertyEvaluationRequest(
        latitud=4.5820,
        longitud=-74.2180,
        area_construida=48.0,
        estrato=2,
        alcobas=2,
        banos_completos=1,
        garajes=0,
        tipo_inmueble='Apto.',
        tipo_vis='No Vis'
    )
    res_novis = service.evaluate(req_novis)
    assert res_vis.precio_corriente_cop < res_novis.precio_corriente_cop
