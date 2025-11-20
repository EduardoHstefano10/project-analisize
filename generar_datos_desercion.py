"""
Script para generar datos sintéticos de deserción académica
según las especificaciones del PDF del proyecto
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Generar 2000 registros como indica el PDF
n_registros = 2000

print(f"Generando {n_registros} registros sintéticos...")

# ===== HÁBITOS Y SALUD =====
sueno_horas = np.random.normal(7, 1.5, n_registros).clip(4, 12)
actividad_fisica = np.random.choice(['Nunca', 'Ocasional', 'Regular', 'Frecuente'], n_registros,
                                   p=[0.15, 0.35, 0.35, 0.15])
alimentacion = np.random.choice(['Deficiente', 'Regular', 'Buena', 'Excelente'], n_registros,
                               p=[0.1, 0.3, 0.4, 0.2])
estilo_vida = np.random.choice(['Sedentario', 'Moderado', 'Activo'], n_registros,
                              p=[0.25, 0.50, 0.25])

# ===== PERSONALES Y EMOCIONALES =====
estres_academico = np.random.choice(['Bajo', 'Moderado', 'Alto', 'Muy Alto'], n_registros,
                                   p=[0.15, 0.35, 0.35, 0.15])
apoyo_familiar = np.random.choice(['Bajo', 'Moderado', 'Alto', 'Muy Alto'], n_registros,
                                 p=[0.10, 0.25, 0.40, 0.25])
bienestar = np.random.choice(['Bajo', 'Moderado', 'Alto', 'Muy Alto'], n_registros,
                            p=[0.10, 0.30, 0.40, 0.20])

# ===== ACADÉMICAS =====
asistencia = np.random.normal(85, 10, n_registros).clip(50, 100)
horas_estudio = np.random.normal(15, 5, n_registros).clip(0, 40)
interes_academico = np.random.choice(['Bajo', 'Moderado', 'Alto', 'Muy Alto'], n_registros,
                                    p=[0.15, 0.30, 0.35, 0.20])
rendimiento_academico = np.random.choice(['Deficiente', 'Regular', 'Bueno', 'Excelente'], n_registros,
                                        p=[0.15, 0.30, 0.40, 0.15])
promedio_acumulado = np.random.normal(14, 2, n_registros).clip(8, 20)

# ===== SOCIOECONÓMICAS =====
carga_laboral = np.random.choice(['No trabaja', 'Tiempo parcial', 'Tiempo completo'], n_registros,
                                p=[0.40, 0.45, 0.15])
beca = np.random.choice(['Sí', 'No'], n_registros, p=[0.35, 0.65])
deudor = np.random.choice(['Sí', 'No'], n_registros, p=[0.25, 0.75])

# ===== VARIABLE OBJETIVO: RIESGO DE DESERCIÓN =====
# Generamos la variable objetivo basada en una combinación de factores

def calcular_riesgo(i):
    """Calcula el riesgo de deserción basado en múltiples factores"""
    score = 0

    # Factor sueño
    if sueno_horas[i] < 5 or sueno_horas[i] > 9:
        score += 1

    # Factor actividad física
    if actividad_fisica[i] == 'Nunca':
        score += 1

    # Factor alimentación
    if alimentacion[i] == 'Deficiente':
        score += 2

    # Factor estilo de vida
    if estilo_vida[i] == 'Sedentario':
        score += 1

    # Factor estrés (importante)
    if estres_academico[i] == 'Muy Alto':
        score += 3
    elif estres_academico[i] == 'Alto':
        score += 2

    # Factor apoyo familiar (importante)
    if apoyo_familiar[i] == 'Bajo':
        score += 3
    elif apoyo_familiar[i] == 'Moderado':
        score += 1

    # Factor bienestar
    if bienestar[i] == 'Bajo':
        score += 2
    elif bienestar[i] == 'Moderado':
        score += 1

    # Factor asistencia (muy importante)
    if asistencia[i] < 70:
        score += 4
    elif asistencia[i] < 80:
        score += 2
    elif asistencia[i] < 85:
        score += 1

    # Factor horas de estudio
    if horas_estudio[i] < 8:
        score += 2
    elif horas_estudio[i] < 12:
        score += 1

    # Factor interés académico (muy importante)
    if interes_academico[i] == 'Bajo':
        score += 3
    elif interes_academico[i] == 'Moderado':
        score += 1

    # Factor rendimiento académico (muy importante)
    if rendimiento_academico[i] == 'Deficiente':
        score += 4
    elif rendimiento_academico[i] == 'Regular':
        score += 2

    # Factor promedio acumulado (muy importante)
    if promedio_acumulado[i] < 11:
        score += 4
    elif promedio_acumulado[i] < 13:
        score += 2
    elif promedio_acumulado[i] < 14:
        score += 1

    # Factor carga laboral
    if carga_laboral[i] == 'Tiempo completo':
        score += 3
    elif carga_laboral[i] == 'Tiempo parcial':
        score += 1

    # Factor beca
    if beca[i] == 'No':
        score += 1

    # Factor deudor (muy importante)
    if deudor[i] == 'Sí':
        score += 3

    # Clasificar según el score
    if score <= 3:
        return 'Sin riesgo'
    elif score <= 7:
        return 'Riesgo leve'
    elif score <= 12:
        return 'Riesgo moderado'
    elif score <= 17:
        return 'Riesgo alto'
    else:
        return 'Riesgo crítico'

# Generar la variable objetivo
riesgo_desercion = [calcular_riesgo(i) for i in range(n_registros)]

# Crear DataFrame
df = pd.DataFrame({
    # Hábitos y salud
    'Sueño_horas': sueno_horas,
    'Actividad_física': actividad_fisica,
    'Alimentación': alimentacion,
    'Estilo_de_vida': estilo_vida,

    # Personales y emocionales
    'Estrés_académico': estres_academico,
    'Apoyo_familiar': apoyo_familiar,
    'Bienestar': bienestar,

    # Académicas
    'Asistencia': asistencia,
    'Horas_estudio': horas_estudio,
    'Interés_académico': interes_academico,
    'Rendimiento_académico': rendimiento_academico,
    'Promedio_acumulado': promedio_acumulado,

    # Socioeconómicas
    'Carga_laboral': carga_laboral,
    'Beca': beca,
    'Deudor': deudor,

    # Variable objetivo
    'Riesgo_deserción': riesgo_desercion
})

# Mostrar distribución de clases
print("\nDistribución de clases:")
print(df['Riesgo_deserción'].value_counts().sort_index())

# Guardar el dataset
output_file = 'datos_desercion_academica.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Dataset guardado en: {output_file}")
print(f"   Total de registros: {len(df)}")
print(f"   Total de características: {len(df.columns) - 1}")
