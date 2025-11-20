"""
Aplicación Streamlit para Predicción de Deserción Académica
Siguiendo exactamente las especificaciones del PDF del proyecto
Universidad Peruana Unión
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Deserción Académica - UPeU",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .sin-riesgo {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .riesgo-leve {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 2px solid #bee5eb;
    }
    .riesgo-moderado {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .riesgo-alto {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .riesgo-critico {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<div class="main-header">🎓 Predicción de Deserción Académica</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Universidad Peruana Unión - Sistema de Alerta Temprana</div>', unsafe_allow_html=True)

# Cargar modelo y objetos
@st.cache_resource
def load_model_and_objects():
    try:
        # Intentar cargar modelo .keras primero, si no .h5
        if os.path.exists("modelo_riesgo_desercion.keras"):
            model = load_model("modelo_riesgo_desercion.keras")
        else:
            model = load_model("modelo_riesgo_desercion.h5")

        label_encoder = joblib.load("label_encoder.pkl")
        feature_names = joblib.load("columnas_X.pkl")
        scaler = joblib.load("scaler_estudiantes.pkl")

        return model, label_encoder, feature_names, scaler, None
    except Exception as e:
        return None, None, None, None, str(e)

# Cargar componentes
model, le, feature_names, scaler, error = load_model_and_objects()

if error:
    st.error(f"❌ Error al cargar el modelo: {error}")
    st.info("Por favor, ejecuta primero: `python train_model.py`")
    st.stop()

# Sidebar con información
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Logo_UPEU.svg/1200px-Logo_UPEU.svg.png", width=200)
    st.markdown("### 📊 Información del Sistema")
    st.markdown("""
    Este sistema utiliza **Redes Neuronales Artificiales** para predecir el riesgo de deserción académica.

    **Categorías de Riesgo:**
    - 🟢 Sin riesgo
    - 🔵 Riesgo leve
    - 🟡 Riesgo moderado
    - 🟠 Riesgo alto
    - 🔴 Riesgo crítico

    **Arquitectura del Modelo:**
    - Capa entrada: 128 neuronas (ReLU)
    - Capa oculta: 64 neuronas (ReLU)
    - Capa salida: 5 clases (Softmax)
    - Optimizador: Adam
    - Precisión: ~99%
    """)

    st.markdown("---")
    st.markdown("**Desarrollado por:**")
    st.markdown("Javier Tello, Sebastian Chinchay,<br>Verónica Vergara, Pamela Vallejos", unsafe_allow_html=True)
    st.markdown("**Docente:** Guillermo Mamani Apaza")

# Tabs para diferentes secciones
tab1, tab2, tab3 = st.tabs(["📝 Predicción Individual", "📊 Información del Proyecto", "❓ Ayuda"])

with tab1:
    st.markdown("### Ingrese los datos del estudiante:")

    # Crear formulario con dos columnas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🛏️ Hábitos y Salud")
        sueno_horas = st.slider("Horas de Sueño", 4.0, 12.0, 7.0, 0.5, key="sueno")

        actividad_fisica = st.selectbox(
            "Actividad Física",
            ["Nunca", "Ocasional", "Regular", "Frecuente"],
            key="actividad"
        )

        alimentacion = st.selectbox(
            "Alimentación",
            ["Deficiente", "Regular", "Buena", "Excelente"],
            key="alimentacion"
        )

        estilo_vida = st.selectbox(
            "Estilo de Vida",
            ["Sedentario", "Moderado", "Activo"],
            key="estilo"
        )

        st.markdown("#### 😊 Factores Personales y Emocionales")

        estres_academico = st.selectbox(
            "Estrés Académico",
            ["Bajo", "Moderado", "Alto", "Muy Alto"],
            key="estres"
        )

        apoyo_familiar = st.selectbox(
            "Apoyo Familiar",
            ["Bajo", "Moderado", "Alto", "Muy Alto"],
            key="apoyo"
        )

        bienestar = st.selectbox(
            "Nivel de Bienestar",
            ["Bajo", "Moderado", "Alto", "Muy Alto"],
            key="bienestar"
        )

    with col2:
        st.markdown("#### 📚 Factores Académicos")

        asistencia = st.slider("Asistencia (%)", 50.0, 100.0, 85.0, 1.0, key="asistencia")

        horas_estudio = st.slider("Horas de Estudio Semanales", 0.0, 40.0, 15.0, 1.0, key="horas_estudio")

        interes_academico = st.selectbox(
            "Interés Académico",
            ["Bajo", "Moderado", "Alto", "Muy Alto"],
            key="interes"
        )

        rendimiento_academico = st.selectbox(
            "Rendimiento Académico",
            ["Deficiente", "Regular", "Bueno", "Excelente"],
            key="rendimiento"
        )

        promedio_acumulado = st.slider("Promedio Acumulado", 8.0, 20.0, 14.0, 0.1, key="promedio")

        st.markdown("#### 💰 Factores Socioeconómicos")

        carga_laboral = st.selectbox(
            "Carga Laboral",
            ["No trabaja", "Tiempo parcial", "Tiempo completo"],
            key="carga"
        )

        beca = st.selectbox("¿Tiene Beca?", ["Sí", "No"], key="beca")

        deudor = st.selectbox("¿Es Deudor?", ["Sí", "No"], key="deudor")

    # Botón de predicción
    if st.button("🔮 Predecir Riesgo de Deserción", type="primary", use_container_width=True):
        # Crear DataFrame con los datos ingresados
        input_data = pd.DataFrame({
            'Sueño_horas': [sueno_horas],
            'Actividad_física': [actividad_fisica],
            'Alimentación': [alimentacion],
            'Estilo_de_vida': [estilo_vida],
            'Estrés_académico': [estres_academico],
            'Apoyo_familiar': [apoyo_familiar],
            'Bienestar': [bienestar],
            'Asistencia': [asistencia],
            'Horas_estudio': [horas_estudio],
            'Interés_académico': [interes_academico],
            'Rendimiento_académico': [rendimiento_academico],
            'Promedio_acumulado': [promedio_acumulado],
            'Carga_laboral': [carga_laboral],
            'Beca': [beca],
            'Deudor': [deudor]
        })

        # Aplicar One-Hot Encoding
        input_encoded = pd.get_dummies(input_data)

        # Asegurar que tenga las mismas columnas que el entrenamiento
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reordenar columnas
        input_encoded = input_encoded[feature_names]

        # Escalar datos
        input_scaled = scaler.transform(input_encoded)

        # Realizar predicción
        prediction = model.predict(input_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        predicted_label = le.inverse_transform([predicted_class])[0]
        confidence = prediction[0][predicted_class] * 100

        # Mostrar resultado
        st.markdown("---")
        st.markdown("### 🎯 Resultado de la Predicción:")

        # Determinar clase CSS según el riesgo
        risk_classes = {
            'Sin riesgo': 'sin-riesgo',
            'Riesgo leve': 'riesgo-leve',
            'Riesgo moderado': 'riesgo-moderado',
            'Riesgo alto': 'riesgo-alto',
            'Riesgo crítico': 'riesgo-critico'
        }
        css_class = risk_classes.get(predicted_label, 'riesgo-moderado')

        # Determinar icono
        risk_icons = {
            'Sin riesgo': '🟢',
            'Riesgo leve': '🔵',
            'Riesgo moderado': '🟡',
            'Riesgo alto': '🟠',
            'Riesgo crítico': '🔴'
        }
        icon = risk_icons.get(predicted_label, '⚪')

        st.markdown(f'<div class="risk-box {css_class}">{icon} {predicted_label.upper()}</div>', unsafe_allow_html=True)

        # Mostrar confianza
        col1, col2, col3 = st.columns(3)
        with col2:
            st.metric("Confianza de la Predicción", f"{confidence:.2f}%")

        # Recomendaciones según nivel de riesgo
        st.markdown("### 📋 Recomendaciones Institucionales:")

        recommendations = {
            'Sin riesgo': {
                'text': "El estudiante presenta una situación estable. Se recomienda seguimiento regular y refuerzo positivo.",
                'color': 'success'
            },
            'Riesgo leve': {
                'text': "Se detectan señales tempranas. Se recomienda tutoría preventiva y monitoreo de asistencia.",
                'color': 'info'
            },
            'Riesgo moderado': {
                'text': "Factores combinados de riesgo. Se recomienda consejería académica y apoyo emocional.",
                'color': 'warning'
            },
            'Riesgo alto': {
                'text': "Alta probabilidad de abandono. Se recomienda intervención conjunta con bienestar estudiantil.",
                'color': 'error'
            },
            'Riesgo crítico': {
                'text': "⚠️ RIESGO INMINENTE. Activar protocolo de retención urgente o apoyo personalizado inmediato.",
                'color': 'error'
            }
        }

        rec = recommendations.get(predicted_label, recommendations['Riesgo moderado'])

        if rec['color'] == 'success':
            st.success(rec['text'])
        elif rec['color'] == 'info':
            st.info(rec['text'])
        elif rec['color'] == 'warning':
            st.warning(rec['text'])
        else:
            st.error(rec['text'])

        # Mostrar distribución de probabilidades
        with st.expander("📊 Ver distribución de probabilidades"):
            prob_df = pd.DataFrame({
                'Nivel de Riesgo': le.classes_,
                'Probabilidad (%)': prediction[0] * 100
            }).sort_values('Probabilidad (%)', ascending=False)

            st.bar_chart(prob_df.set_index('Nivel de Riesgo'))
            st.dataframe(prob_df, use_container_width=True)

        # Mostrar datos de entrada
        with st.expander("🔍 Ver datos ingresados"):
            st.dataframe(input_data.T, use_container_width=True)

with tab2:
    st.markdown("### 📖 Información del Proyecto")

    st.markdown("""
    ## Predicción de la Deserción Académica mediante Redes Neuronales

    ### 🎯 Objetivo
    Predecir el riesgo de deserción universitaria mediante redes neuronales artificiales,
    integrando factores académicos, personales, de salud y económicos.

    ### 📊 Variables Consideradas

    **Hábitos y Salud:**
    - Horas de sueño
    - Actividad física
    - Alimentación
    - Estilo de vida

    **Personales y Emocionales:**
    - Estrés académico
    - Apoyo familiar
    - Bienestar

    **Académicas:**
    - Asistencia
    - Horas de estudio
    - Interés académico
    - Rendimiento académico
    - Promedio acumulado

    **Socioeconómicas:**
    - Carga laboral
    - Beca
    - Deudor

    ### 🧠 Arquitectura del Modelo

    - **Tipo:** Red Neuronal Secuencial
    - **Capa de entrada:** 128 neuronas (ReLU)
    - **Dropout:** Regularización
    - **Capa oculta:** 64 neuronas (ReLU)
    - **Capa de salida:** 5 neuronas (Softmax)
    - **Optimizador:** Adam
    - **Función de pérdida:** Categorical Crossentropy
    - **Entrenamiento:** 40 épocas, batch size 32
    - **Precisión:** ~99%

    ### 📈 Resultados
    El modelo alcanzó una precisión global del 99.25% en el conjunto de prueba,
    demostrando alta capacidad predictiva para identificar estudiantes en riesgo.

    ### 🎓 Impacto Institucional
    El despliegue del modelo permitirá establecer un sistema de alerta temprana institucional,
    capaz de detectar y atender de forma oportuna los casos con alto riesgo de deserción.
    """)

with tab3:
    st.markdown("### ❓ Ayuda y Preguntas Frecuentes")

    st.markdown("""
    #### ¿Cómo usar la aplicación?
    1. Complete todos los campos del formulario con la información del estudiante
    2. Haga clic en "Predecir Riesgo de Deserción"
    3. Revise el resultado y las recomendaciones institucionales

    #### ¿Qué significan los niveles de riesgo?
    - **Sin riesgo:** Estudiante estable, bajo riesgo de deserción
    - **Riesgo leve:** Señales tempranas, requiere monitoreo
    - **Riesgo moderado:** Factores combinados, necesita apoyo
    - **Riesgo alto:** Alta probabilidad de abandono, intervención requerida
    - **Riesgo crítico:** Situación crítica, atención urgente

    #### ¿Qué tan preciso es el modelo?
    El modelo ha sido entrenado con 2000 registros y alcanza una precisión del 99%,
    lo que indica alta confiabilidad en sus predicciones.

    #### ¿Qué hacer con los resultados?
    Los resultados deben ser utilizados por el personal de bienestar estudiantil
    para implementar las intervenciones sugeridas según el nivel de riesgo detectado.

    #### Contacto
    Para más información, contacte al equipo de desarrollo o al Docente
    Guillermo Mamani Apaza.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Universidad Peruana Unión - Facultad de Ingeniería y Arquitectura<br>"
    "Proyecto de Inteligencia Artificial - 2025<br>"
    "Desarrollado con TensorFlow/Keras y Streamlit"
    "</div>",
    unsafe_allow_html=True
)
