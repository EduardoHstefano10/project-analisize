"""
Script rápido para verificar que el modelo carga correctamente
"""
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

print("Cargando modelo y componentes...")
model = load_model("modelo_estudiantes.keras")
scaler = joblib.load("scaler_estudiantes.pkl")
feature_names = joblib.load("feature_names.pkl")

print("✓ Modelo cargado exitosamente")
print("✓ Scaler cargado exitosamente")
print("✓ Feature names cargado exitosamente")

# Crear datos de prueba
test_data = pd.DataFrame({
    'Promedio_ponderado': [15.0],
    'Creditos_matriculados': [20.0],
    'Porcentaje_creditos_aprobados': [75.0],
    'Cursos_desaprobados': [1.0],
    'Asistencia': [90.0],
    'Retiros_cursos': [1.0],
    'Edad': [20],
    'Horas_trabajo_semana': [15.0],
    'Anio_ingreso': [2020],
    'Numero_ciclos_academicos': [8.0],
    'Cursos_matriculados_ciclo': [6.0],
    'Horas_estudio_semana': [15.0],
    'indice_regularidad': [65.0],
    'Intentos_aprobacion_curso': [0.0]
})

print("\nProbando predicción...")
test_scaled = scaler.transform(test_data)
prediction = model.predict(test_scaled, verbose=0)[0][0]

print(f"✓ Predicción exitosa: {prediction:.2f}")
print("\n¡Todo funciona correctamente! 🎉")
