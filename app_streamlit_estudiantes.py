"""
Aplicación Streamlit para predicción de rendimiento académico
usando Redes Neuronales Artificiales (RNA)
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Rendimiento Académico",
    page_icon="🎓",
    layout="centered"
)

# Título de la aplicación
st.title("🎓 Predicción de Rendimiento Académico")
st.markdown("### Usando Redes Neuronales Artificiales (RNA)")
st.markdown("---")

# Cargar modelo, scaler y metadata
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model("modelo_estudiantes.keras")
        scaler = joblib.load("scaler_estudiantes.pkl")
        feature_names = joblib.load("feature_names.pkl")

        metadata = {}
        if os.path.exists("metadata.json"):
            with open("metadata.json", 'r') as f:
                metadata = json.load(f)

        return model, scaler, feature_names, metadata
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.info("Por favor, ejecuta primero `train_model.py` para entrenar el modelo.")
        return None, None, None, None

# Cargar componentes
model, scaler, feature_names, metadata = load_model_and_scaler()

if model is not None:
    # Mostrar información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write(f"**Características:** {metadata.get('n_features', 'N/A')}")
        st.write(f"**MAE en entrenamiento:** {metadata.get('train_mae', 'N/A'):.4f}")
        st.write(f"**MAE en prueba:** {metadata.get('test_mae', 'N/A'):.4f}")
        st.write(f"**MSE en entrenamiento:** {metadata.get('train_mse', 'N/A'):.4f}")
        st.write(f"**MSE en prueba:** {metadata.get('test_mse', 'N/A'):.4f}")

    st.markdown("---")
    st.markdown("### 📝 Ingrese los datos del estudiante:")

    # Formulario de entrada
    with st.form("prediccion_form"):
        col1, col2 = st.columns(2)

        with col1:
            promedio_ponderado = st.number_input(
                "Promedio Ponderado",
                min_value=0.0,
                max_value=20.0,
                value=15.0,
                step=0.1
            )

            creditos_matriculados = st.number_input(
                "Créditos Matriculados",
                min_value=0.0,
                max_value=30.0,
                value=20.0,
                step=1.0
            )

            porcentaje_creditos_aprobados = st.number_input(
                "Porcentaje Créditos Aprobados (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

            cursos_desaprobados = st.number_input(
                "Cursos Desaprobados",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=1.0
            )

            asistencia = st.number_input(
                "Asistencia (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0,
                step=1.0
            )

            retiros_cursos = st.number_input(
                "Retiros de Cursos",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=1.0
            )

            edad = st.number_input(
                "Edad",
                min_value=16,
                max_value=50,
                value=20,
                step=1
            )

        with col2:
            horas_trabajo_semana = st.number_input(
                "Horas de Trabajo por Semana",
                min_value=0.0,
                max_value=60.0,
                value=15.0,
                step=1.0
            )

            anio_ingreso = st.number_input(
                "Año de Ingreso",
                min_value=2010,
                max_value=2025,
                value=2020,
                step=1
            )

            numero_ciclos_academicos = st.number_input(
                "Número de Ciclos Académicos",
                min_value=1.0,
                max_value=20.0,
                value=8.0,
                step=1.0
            )

            cursos_matriculados_ciclo = st.number_input(
                "Cursos Matriculados por Ciclo",
                min_value=1.0,
                max_value=10.0,
                value=6.0,
                step=1.0
            )

            horas_estudio_semana = st.number_input(
                "Horas de Estudio por Semana",
                min_value=0.0,
                max_value=60.0,
                value=15.0,
                step=1.0
            )

            indice_regularidad = st.number_input(
                "Índice de Regularidad",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0
            )

            intentos_aprobacion_curso = st.number_input(
                "Intentos de Aprobación de Curso",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.5
            )

        # Botón de predicción
        submitted = st.form_submit_button("🔮 Predecir Nota Promedio", use_container_width=True)

    if submitted:
        # Crear DataFrame con los datos ingresados
        input_data = pd.DataFrame({
            'Promedio_ponderado': [promedio_ponderado],
            'Creditos_matriculados': [creditos_matriculados],
            'Porcentaje_creditos_aprobados': [porcentaje_creditos_aprobados],
            'Cursos_desaprobados': [cursos_desaprobados],
            'Asistencia': [asistencia],
            'Retiros_cursos': [retiros_cursos],
            'Edad': [edad],
            'Horas_trabajo_semana': [horas_trabajo_semana],
            'Anio_ingreso': [anio_ingreso],
            'Numero_ciclos_academicos': [numero_ciclos_academicos],
            'Cursos_matriculados_ciclo': [cursos_matriculados_ciclo],
            'Horas_estudio_semana': [horas_estudio_semana],
            'indice_regularidad': [indice_regularidad],
            'Intentos_aprobacion_curso': [intentos_aprobacion_curso]
        })

        # Escalar los datos
        input_scaled = scaler.transform(input_data)

        # Realizar predicción
        prediction = model.predict(input_scaled, verbose=0)[0][0]

        # Mostrar resultado
        st.markdown("---")
        st.markdown("### 🎯 Resultado de la Predicción:")

        # Crear columnas para el resultado
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.metric(
                label="Nota Promedio Predicha",
                value=f"{prediction:.2f}",
                delta=None
            )

            # Mostrar nivel de rendimiento
            if prediction >= 16:
                st.success("🌟 Excelente rendimiento")
            elif prediction >= 14:
                st.info("👍 Buen rendimiento")
            elif prediction >= 11:
                st.warning("⚠️ Rendimiento regular")
            else:
                st.error("❌ Rendimiento bajo")

        # Mostrar detalles
        with st.expander("📊 Ver detalles de entrada"):
            st.dataframe(input_data.T, use_container_width=True)

else:
    st.warning("⚠️ No se pudo cargar el modelo. Por favor, entrena el modelo primero.")
    st.code("python train_model.py", language="bash")

# Footer
st.markdown("---")
st.markdown("**Desarrollado con:**")
st.markdown("🧠 TensorFlow/Keras | 📊 Streamlit | 🐍 Python")
