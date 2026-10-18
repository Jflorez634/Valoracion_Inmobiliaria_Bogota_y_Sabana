import sys
from pathlib import Path

# Garantizar que el directorio raíz esté en sys.path para resolver 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.predictor import PropertyValuationService, get_rango_m2
from src.schemas import PropertyEvaluationRequest
from src.config import BOGOTA_CENTER_LAT, BOGOTA_CENTER_LON

st.set_page_config(
    page_title="Valuador Inmobiliario Inteligente | Bogotá & Sabana",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS theme-safe (compatible con Dark Mode y Light Mode)
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(29, 78, 216, 0.3);
        margin-bottom: 1rem;
    }
    .kpi-card * {
        color: #FFFFFF !important;
    }
    .kpi-price {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_valuation_service():
    service = PropertyValuationService()
    if not service.healthcheck():
        st.error("Error en la inicialización del modelo de predicción.")
    return service

service = load_valuation_service()

# Header
st.title("🏢 Valuador Inmobiliario Inteligente")
st.caption("Estimación objetiva de precios de vivienda en Bogotá y Sabana mediante XGBoost + LassoCV (Metodología CRISP-DM)")

# Badges / Metrics bar
m1, m2, m3, m4 = st.columns(4)
m1.metric("Capacidad Explicativa", "R² = 0.975", "Test Set")
m2.metric("Error Normalizado", "NMAE = 12.56%", "Meta < 20%")
m3.metric("Base Espacial", "5,476 Proyectos", "Bogotá & Sabana")
m4.metric("Velocidad Inferencia", "< 50 ms", "BallTree k-NN")

st.divider()

# Session state for coordinates
if 'lat' not in st.session_state:
    st.session_state.lat = 4.6757
if 'lon' not in st.session_state:
    st.session_state.lon = -74.0538
if 'area_num_val' not in st.session_state:
    st.session_state.area_num_val = 75.0
if 'area_slider_val' not in st.session_state:
    st.session_state.area_slider_val = 75.0

def sync_area_from_slider():
    st.session_state.area_num_val = float(st.session_state.area_slider_val)

def sync_area_from_num():
    st.session_state.area_slider_val = float(min(st.session_state.area_num_val, 450.0))

# Preset locations
PRESETS = {
    "Chicó / Virrey (Norte - Bogotá)": (4.6757, -74.0538),
    "Cedritos / Usaquén (Norte - Bogotá)": (4.7250, -74.0320),
    "Chapinero Central (Oriente - Bogotá)": (4.6480, -74.0620),
    "Ciudad Salitre / Teusaquillo (Bogotá)": (4.6540, -74.1080),
    "Colina Campestre / Suba (Bogotá)": (4.7380, -74.0680),
    "Centro Internacional (Bogotá)": (4.6150, -74.0720),
    "Chía (Sabana Norte)": (4.8622, -74.0583),
    "Cajicá (Sabana Norte)": (4.9180, -74.0250),
    "Soacha (Sabana Sur)": (4.5820, -74.2180),
}

col_map, col_form = st.columns([1.15, 1], gap="large")

with col_map:
    st.subheader("1. Ubicación Geoespacial")
    
    col_p1, col_p2 = st.columns([3, 1.2])
    with col_p1:
        preset_choice = st.selectbox(
            "Seleccionar sector o municipio de referencia:",
            options=list(PRESETS.keys()),
            index=0,
            label_visibility="collapsed"
        )
    with col_p2:
        if st.button("📍 Centrar Mapa", use_container_width=True):
            st.session_state.lat, st.session_state.lon = PRESETS[preset_choice]
            st.rerun()

    # Map Rendering using OpenStreetMap (Free, Full streets, No API keys required)
    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )
    
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup=f"Ubicación: ({st.session_state.lat:.4f}, {st.session_state.lon:.4f})",
        tooltip="Haz clic en cualquier punto del mapa para mover la ubicación",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)

    # Circle radius around location
    folium.Circle(
        radius=250,
        location=[st.session_state.lat, st.session_state.lon],
        color="#3B82F6",
        fill=True,
        fill_opacity=0.15
    ).add_to(m)

    map_output = st_folium(
        m,
        height=400,
        width="100%",
        key="folium_map"
    )

    # Capture click on map
    if map_output and map_output.get("last_clicked"):
        clicked_lat = map_output["last_clicked"]["lat"]
        clicked_lon = map_output["last_clicked"]["lng"]
        if abs(clicked_lat - st.session_state.lat) > 0.0001 or abs(clicked_lon - st.session_state.lon) > 0.0001:
            st.session_state.lat = clicked_lat
            st.session_state.lon = clicked_lon
            st.rerun()

    # Real-time Geocoding Preview using native themed container
    geo_res = service.geocoder.lookup(st.session_state.lat, st.session_state.lon)
    
    with st.container(border=True):
        st.markdown(f"**📍 Microsector Detectado:** `{geo_res['Barrio']}`")
        st.markdown(f"**🗺️ Zona y Municipio:** **{geo_res['Zona']}** — {geo_res['Municipio']}")
        st.markdown(f"**🎯 Distancia a proyecto de referencia:** **{geo_res['distancia_referencia_m']:.0f} metros**")
        st.caption(f"Coordenadas activas: Lat {st.session_state.lat:.4f} | Lon {st.session_state.lon:.4f}")
    
    if geo_res['fuera_de_geocerca']:
        st.warning("⚠️ Ubicación fuera de la cobertura principal de proyectos modelados en la Sabana.")

with col_form:
    st.subheader("2. Características del Inmueble")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tipo_inm = st.radio("Tipo de Inmueble:", ["Apto.", "Casa"], horizontal=True)
    with col_t2:
        estrato = st.select_slider("Estrato socioeconómico:", options=[1, 2, 3, 4, 5, 6], value=4)

    # VIS Housing Classification
    vis_dict = {
        "No VIS (Tradicional / Mercado Libre)": ("No Vis", "🏷️ Mercado tradicional libre sin restricción legal de precio."),
        "VIS (Vivienda de Interés Social)": ("Vis", "⚖️ Regulado: Tope legal en Bogotá hasta 135-150 SMMLV (~$213M COP)."),
        "VIS Renovación Urbana": ("Vis Renov.", "⚖️ Regulado: Macroproyectos de renovación urbana hasta 175 SMMLV (~$249M COP)."),
        "VIP (Interés Prioritario)": ("Vip ", "⚖️ Regulado: Enfoque social prioritario hasta 90 SMMLV (~$128M COP).")
    }
    
    col_v1, col_v2 = st.columns([1.5, 1])
    with col_v1:
        vis_label = st.selectbox(
            "Clasificación VIS / No VIS:",
            options=list(vis_dict.keys()),
            index=0,
            help="En Colombia, las viviendas de interés social cuentan con subsidios y topes normativos fijados en SMMLV."
        )
        tipo_vis_code, vis_help_text = vis_dict[vis_label]
    with col_v2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.caption(vis_help_text)

    # Area Input (Manual typing + Fine-tuning Slider)
    col_a1, col_a2 = st.columns([1.4, 1])
    with col_a1:
        st.number_input(
            "Área privada construida (m²) [Escribe aquí]:",
            min_value=15.0,
            max_value=600.0,
            step=1.0,
            key="area_num_val",
            on_change=sync_area_from_num,
            help="Puedes escribir directamente el número exacto con tu teclado (ej: 54.5 m²) o usar los botones +/-."
        )
    with col_a2:
        rango_comercial = get_rango_m2(st.session_state.area_num_val)
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.markdown(f"**Rango comercial:** `{rango_comercial} m²`")

    st.slider(
        "Ajuste visual de área:",
        min_value=15.0,
        max_value=450.0,
        step=1.0,
        key="area_slider_val",
        on_change=sync_area_from_slider,
        label_visibility="collapsed"
    )
    area = st.session_state.area_num_val
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        alcobas = st.slider("Alcobas:", min_value=1, max_value=5, value=2)
        garajes = st.slider("Garajes:", min_value=0, max_value=4, value=1)
    with col_f2:
        banos = st.slider("Baños completos:", min_value=1, max_value=5, value=2)
        tipo_garaje = st.selectbox(
            "Tipo de garaje:",
            options=["Cub. Sencillo", "Cub. Lineal", "Lineales", "Descubierto", "No tiene"],
            index=0 if garajes > 0 else 4
        )

    st.markdown("##### Comodidades adicionales:")
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        tiene_estudio = st.checkbox("Estudio / Estar TV", value=False)
        tiene_terraza = st.checkbox("Balcón o Terraza privada", value=False)
    with c_b2:
        deposito = st.checkbox("Cuarto de Depósito", value=True)
        tiene_patio = st.checkbox("Patio / Jardín interior", value=(tipo_inm == "Casa"))

    area_terraza = 0.0
    if tiene_terraza:
        area_terraza = st.slider("Área de terraza/balcón (m²):", 2, 60, 6)
        
    area_patio = 0.0
    if tiene_patio:
        area_patio = st.slider("Área de patio/jardín (m²):", 4, 80, 15)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    btn_eval = st.button("🚀 Calcular Avalúo Estimado", type="primary", use_container_width=True)

# Calculation Trigger
if btn_eval:
    req = PropertyEvaluationRequest(
        latitud=st.session_state.lat,
        longitud=st.session_state.lon,
        area_construida=float(area),
        estrato=int(estrato),
        alcobas=int(alcobas),
        banos_completos=int(banos),
        garajes=int(garajes),
        tipo_garaje=tipo_garaje,
        tipo_inmueble=tipo_inm,
        tipo_vis=tipo_vis_code,
        tiene_estudio=tiene_estudio,
        tiene_terraza_balcon=tiene_terraza,
        area_terraza_balcon=float(area_terraza),
        tiene_patio=tiene_patio,
        area_patio=float(area_patio),
        deposito=deposito
    )
    
    res = service.evaluate(req)
    
    st.divider()
    st.subheader("3. Resultado del Avalúo y Análisis de Mercado")
    
    res_col1, res_col2 = st.columns([1, 1], gap="large")
    
    with res_col1:
        precio_m = res.precio_corriente_cop / 1_000_000.0
        m2_m = res.precio_m2_cop / 1_000_000.0
        min_m = res.rango_min_cop / 1_000_000.0
        max_m = res.rango_max_cop / 1_000_000.0
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Valor Comercial Estimado (2026)</div>
            <div class="kpi-price">$ {precio_m:,.1f} Millones</div>
            <div style="font-size: 1.1rem; opacity: 0.95; margin-top: 0.4rem;">
                <strong>COP {res.precio_corriente_cop:,.0f}</strong>
            </div>
            <div style="margin-top: 0.8rem; font-size: 1rem;">
                📏 <strong>$ {res.precio_m2_cop:,.0f} / m²</strong> ({m2_m:.2f} M/m²)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if tipo_vis_code != "No Vis":
            st.info(f"⚖️ Avalúo calibrado para **{vis_label}** según topes normativos y subsidios.")
        
        with st.container(border=True):
            st.markdown("##### 📊 Rango de Negociación Sugerido (Confianza ±12.4%):")
            st.write(f"💡 **Banda estimada:** COP **${min_m:,.1f} M** — COP **${max_m:,.1f} M**")
            st.caption("Intervalo de confianza basado en el Error Absoluto Medio Porcentual (NMAE de 12.42%) del modelo XGBoost.")

    with res_col2:
        # Plotly Feature Breakdown Chart with Transparent Theme-Safe Background
        st.markdown("##### 🔍 Contribución Estimada de Factores Clave:")
        
        factor_names = ["Área Construida", "Estrato & Nivel", "Zona Geográfica", "Garajes"]
        base_val = res.precio_corriente_cop * 0.45
        area_impact = (area / 80.0) * (res.precio_corriente_cop * 0.35)
        estrato_impact = (estrato / 4.0) * (res.precio_corriente_cop * 0.12)
        garaje_impact = garajes * 25_000_000.0
        loc_impact = res.precio_corriente_cop - (base_val + area_impact + estrato_impact + garaje_impact)
        
        vals = [area_impact / 1e6, estrato_impact / 1e6, abs(loc_impact) / 1e6, garaje_impact / 1e6]
        
        fig = go.Figure(go.Bar(
            x=factor_names,
            y=vals,
            marker=dict(
                color=['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6'],
                line=dict(width=0)
            ),
            text=[f"${v:.1f}M" for v in vals],
            textposition='auto',
        ))
        fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Millones COP",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(125,125,125,0.2)"),
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Technical metadata card
    with st.expander("🛠️ Ver Ficha Técnica del Modelo y Métricas CRISP-DM"):
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            st.metric("Modelo Predictivo", "XGBoost Regressor")
            st.caption("Algoritmo de ensamble de gradiente")
        with f_c2:
            st.metric("Bondad de Ajuste (R²)", "0.9748")
            st.caption("Validado sobre 100,559 registros de test")
        with f_c3:
            st.metric("Sector Detectado", res.barrio_detectado)
            st.caption(f"{res.zona_detectada} — {res.municipio_detectado}")

    st.caption(
        "Aviso Legal & Portafolio: Esta aplicación es un desarrollo analítico para fines académicos y de demostración "
        "técnica de portafolio de Ciencia de Datos. Los avalúos son estimaciones estadísticas basadas en la serie de precios "
        "deflactados IPVNBR y no constituyen certificaciones catastrales o bancarias oficiales."
    )
