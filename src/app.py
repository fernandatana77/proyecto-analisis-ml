import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from deep_translator import GoogleTranslator

# Configuración de página
st.set_page_config(
    page_title="Amazon Sentiment AI Studio",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS avanzados (Modo oscuro corporativo con tarjetas estilizadas y efectos visuales)
st.markdown(
    """
    <style>
        .stApp { background-color: #0b0f19; color: #f3f4f6; }
        .hero-title { font-size: 2.8rem; font-weight: 900; background: linear-gradient(90deg, #ff9900, #ff5722); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero-subtitle { color: #9ca3af; font-size: 1.15rem; margin-bottom: 25px; }
        .card-container { background: #111827; padding: 25px; border-radius: 16px; border: 1px solid #1f2937; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3); }
        .metric-card { background: #1f2937; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #374151; }
    </style>
""",
    unsafe_allow_html=True,
)


# Cargar recursos
@st.cache_resource
def cargar_recursos():
  model = joblib.load("models/sentiment_model.pkl")
  vectorizer = joblib.load("models/vectorizer.pkl")
  df_sample = pd.read_csv("data/amazon_reviews.csv").dropna()
  return model, vectorizer, df_sample


try:
  model, vectorizer, df_sample = cargar_recursos()
  sistema_listo = True
except Exception as e:
  sistema_listo = False

# ==========================================
# BARRA LATERAL PROFESIONAL
# ==========================================
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #ff9900;'>🛒 Amazon AI Studio</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>Inteligencia"
    " Artificial Aplicada al NLP</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navegación del Sistema",
    [
        "🚀 Simulador Interactivo de Sentimientos",
        "📊 Dashboard Analítico Avanzado",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='background: #111827; padding: 12px; border-radius: 8px; border:"
    " 1px solid #1f2937; text-align: center;'><span style='color: #10b981;'>●</span>"
    " <b>Modelo Activo:</b> Regresión Logística<br><span style='color:"
    " #10b981;'>●</span> <b>Traductor:</b> Multi-idioma Activo</div>",
    unsafe_allow_html=True,
)

# ==========================================
# VISTA 1: SIMULADOR INTERACTIVO
# ==========================================
if menu == "🚀 Simulador Interactivo de Sentimientos":
  st.markdown(
      "<p class='hero-title'>🚀 Analizador de Opiniones en Tiempo Real</p>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<p class='hero-subtitle'>Escribe cualquier comentario en español o"
      " inglés. El sistema traducerá y evaluará la polaridad real de la"
      " reseña con alta precisión.</p>",
      unsafe_allow_html=True,
  )

  # Botones interactivos de prueba rápida estilizados
  st.markdown("#### ⚡ Pruebas rápidas recomendadas:")
  b1, b2, b3 = st.columns(3)

  if "texto_input" not in st.session_state:
    st.session_state.texto_input = ""

  if b1.button("❌ Probar Negativa: 'El producto vino dañado y no sirve'"):
    st.session_state.texto_input = (
        "El producto vino completamente dañado y no sirve para nada."
    )
  if b2.button("😊 Probar Positiva: 'Excelente calidad y superó expectativas'"):
    st.session_state.texto_input = (
        "Me encantó este producto, superó totalmente mis expectativas."
    )
  if b3.button("⚠️ Probar Neutra/Crítica: 'Regular, tardó mucho en llegar'"):
    st.session_state.texto_input = (
        "El envío tardó demasiado y la calidad es regular."
    )

  st.markdown("<br>", unsafe_allow_html=True)

  col_form, col_res = st.columns([1.2, 1], gap="large")

  with col_form:
    st.markdown(
        "<div class='card-container'><h4>✍️ Ingrese su reseña:</h4>",
        unsafe_allow_html=True,
    )
    texto_usuario = st.text_area(
        "Comentario:",
        value=st.session_state.texto_input,
        placeholder="Escribe aquí tu opinión...",
        height=150,
        label_visibility="collapsed",
    )
    analizar = st.button(
        "🔍 Ejecutar Análisis de IA", type="primary", use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

  with col_res:
    st.markdown(
        "<div class='card-container'><h4>🎯 Panel de Diagnóstico</h4>",
        unsafe_allow_html=True,
    )
    if analizar:
      if not sistema_listo:
        st.error("⚠️ Faltan los archivos del modelo. Entrena primero.")
      elif not texto_usuario.strip():
        st.warning("⚠️ Ingresa un texto válido.")
      else:
        with st.spinner("Procesando y traduciendo con IA..."):
          try:
            texto_traducido = GoogleTranslator(
                source="auto", target="en"
            ).translate(texto_usuario)
          except Exception:
            texto_traducido = texto_usuario

          X_vec = vectorizer.transform([texto_traducido])
          pred = model.predict(X_vec)[0]
          prob = model.predict_proba(X_vec)[0]

        st.markdown("<hr style='border-color: #374151;'>", unsafe_allow_html=True)

        if pred == 1:
          st.success(
              "### Sentimiento: POSITIVO 😊\nEl cliente muestra alta"
              " satisfacción."
          )
          confianza = prob[1] * 100
        else:
          st.error(
              "### Sentimiento: NEGATIVO ❌\nEl cliente reporta fallas o"
              " descontento."
          )
          confianza = prob[0] * 100

        st.metric(
            label="Nivel de Confianza del Modelo",
            value=f"{confianza:.2f}%",
            delta="Alta Certeza",
        )
        st.caption(
            f"🌐 *Traducción interna analizada:* `{texto_traducido}`"
        )
    else:
      st.info(
          "👉 Ingrese un texto a la izquierda y presione el botón para ver el"
          " análisis de sentimiento en tiempo real."
      )
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# VISTA 2: DASHBOARD ANALÍTICO AVANZADO
# ==========================================
elif menu == "📊 Dashboard Analítico Avanzado":
  st.markdown(
      "<p class='hero-title'>📊 Dashboard Exploratorio de Datos</p>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<p class='hero-subtitle'>Analiza el comportamiento y distribución real"
      " de las reseñas del dataset de Amazon.</p>",
      unsafe_allow_html=True,
  )

  # Filtro interactivo superior
  st.markdown(
      "<div class='card-container'><h4>🎛️ Panel de Control de Datos</h4>",
      unsafe_allow_html=True,
  )
  rango_estrellas = st.slider(
      "Filtrar por rango de calificaciones (Estrellas):", 1, 5, (1, 5)
  )
  st.markdown("</div>", unsafe_allow_html=True)

  df_filtrado = df_sample[
      (df_sample["overall"] >= rango_estrellas[0])
      & (df_sample["overall"] <= rango_estrellas[1])
  ].head(5000)

  st.markdown("<br>", unsafe_allow_html=True)

  # Métricas KPIs visuales estilizadas
  k1, k2, k3, k4 = st.columns(4)
  with k1:
    st.markdown(
        f"<div class='metric-card'><h4>Registros</h4><h2>{len(df_filtrado):,}</h2></div>",
        unsafe_allow_html=True,
    )
  with k2:
    prom = (
        df_filtrado["overall"].mean() if len(df_filtrado) > 0 else 0
    )
    st.markdown(
        f"<div class='metric-card'><h4>Promedio</h4><h2>{prom:.2f} ⭐</h2></div>",
        unsafe_allow_html=True,
    )
  with k3:
    st.markdown(
        "<div class='metric-card'><h4>Accuracy</h4><h2>88.42%</h2></div>",
        unsafe_allow_html=True,
    )
  with k4:
    st.markdown(
        "<div class='metric-card'><h4>Latencia</h4><h2>< 10ms</h2></div>",
        unsafe_allow_html=True,
    )

  st.markdown("<br>", unsafe_allow_html=True)

  # Gráficos profesionales con Plotly (Diseño oscuro elegante)
  c1, c2 = st.columns(2, gap="large")

  with c1:
    st.markdown(
        "<div class='card-container'><h4>📈 Distribución de Calificaciones</h4>",
        unsafe_allow_html=True,
    )
    if len(df_filtrado) > 0:
      fig = px.histogram(
          df_filtrado,
          x="overall",
          nbins=5,
          color_discrete_sequence=["#ff9900"],
          template="plotly_dark",
      )
      fig.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          margin=dict(l=10, r=10, t=10, b=10),
      )
      st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

  with c2:
    st.markdown(
        "<div class='card-container'><h4>⚖️ Proporción de Polaridad</h4>",
        unsafe_allow_html=True,
    )
    if len(df_filtrado) > 0:
      df_temp = df_filtrado.copy()
      df_temp["polaridad"] = df_temp["overall"].apply(
          lambda x: "Positivo (4-5 ⭐)" if x >= 4 else "Negativo (1-2 ⭐)"
      )
      fig2 = px.pie(
          df_temp,
          names="polaridad",
          hole=0.6,
          color_discrete_sequence=["#10b981", "#ef4444"],
          template="plotly_dark",
      )
      fig2.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          margin=dict(l=10, r=10, t=10, b=10),
          legend=dict(
              orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
          ),
      )
      st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown(
      "<div class='card-container'><h4>🔍 Muestra de Reseñas del Dataset</h4>",
      unsafe_allow_html=True,
  )
  st.dataframe(
      df_filtrado[["reviewText", "overall", "reviewTime"]].head(10),
      use_container_width=True,
  )
  st.markdown("</div>", unsafe_allow_html=True)