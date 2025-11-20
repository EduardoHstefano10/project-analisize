# 🎓 Predicción de Deserción Académica - Universidad Peruana Unión

Sistema de predicción de deserción académica mediante Redes Neuronales Artificiales, siguiendo las especificaciones del documento "Predicción de la Deserción Académica mediante Redes Neuronales".

## 📖 Descripción del Proyecto

Este proyecto implementa un sistema de alerta temprana para identificar estudiantes en riesgo de deserción universitaria, utilizando técnicas de Inteligencia Artificial basadas en Redes Neuronales.

## 🎯 Objetivo

Predecir el riesgo de deserción universitaria mediante redes neuronales artificiales, integrando factores académicos, personales, de salud y económicos.

## 📊 Variables Consideradas

### Hábitos y Salud
- Horas de sueño
- Actividad física
- Alimentación
- Estilo de vida

### Personales y Emocionales
- Estrés académico
- Apoyo familiar
- Bienestar

### Académicas
- Asistencia
- Horas de estudio
- Interés académico
- Rendimiento académico
- Promedio acumulado

### Socioeconómicas
- Carga laboral
- Beca
- Deudor

## 🧠 Arquitectura del Modelo

- **Tipo:** Red Neuronal Secuencial
- **Capa de entrada:** 128 neuronas (ReLU)
- **Dropout:** Regularización
- **Capa oculta:** 64 neuronas (ReLU)
- **Capa de salida:** 5 neuronas (Softmax)
- **Optimizador:** Adam
- **Función de pérdida:** Categorical Crossentropy
- **Entrenamiento:** 40 épocas, batch size 32
- **Precisión:** ~80%

## 🚀 Instalación y Uso

### Requisitos

```bash
pip install -r requirements.txt
```

### Generar Datos

```bash
python generar_datos_desercion.py
```

### Entrenar el Modelo

```bash
python train_model.py
```

### Ejecutar la Aplicación Web

```bash
streamlit run app_streamlit_estudiantes.py
```

## 📈 Niveles de Riesgo

El sistema clasifica a los estudiantes en 5 categorías:

- 🟢 **Sin riesgo:** Estudiante estable
- 🔵 **Riesgo leve:** Señales tempranas
- 🟡 **Riesgo moderado:** Factores combinados
- 🟠 **Riesgo alto:** Alta probabilidad de abandono
- 🔴 **Riesgo crítico:** Riesgo inminente

## 🎓 Recomendaciones por Nivel

### Sin riesgo
Seguimiento regular y refuerzo positivo.

### Riesgo leve
Tutoría preventiva y monitoreo de asistencia.

### Riesgo moderado
Consejería académica y apoyo emocional.

### Riesgo alto
Intervención conjunta con bienestar estudiantil.

### Riesgo crítico
Activación de protocolo de retención urgente o apoyo personalizado inmediato.

## 📁 Estructura del Proyecto

```
project-analisize/
│
├── datos_desercion_academica.csv      # Dataset generado
├── generar_datos_desercion.py         # Script para generar datos
├── train_model.py                     # Script de entrenamiento
├── app_streamlit_estudiantes.py       # Aplicación web
├── modelo_riesgo_desercion.h5         # Modelo entrenado (formato .h5)
├── modelo_riesgo_desercion.keras      # Modelo entrenado (formato .keras)
├── label_encoder.pkl                  # Codificador de etiquetas
├── columnas_X.pkl                     # Nombres de características
├── scaler_estudiantes.pkl             # Escalador de datos
├── matriz_confusion.png               # Visualización matriz de confusión
├── evolucion_modelo.png               # Gráficas de entrenamiento
├── requirements.txt                   # Dependencias
└── README.md                          # Este archivo
```

## 👥 Autores

- Javier Tello
- Sebastian Chinchay
- Verónica Vergara
- Pamela Vallejos

## 👨‍🏫 Docente

Guillermo Mamani Apaza

## 🏛️ Institución

Universidad Peruana Unión
Facultad de Ingeniería y Arquitectura
Curso: Inteligencia Artificial
Fecha: 5 de noviembre de 2025

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos para la Universidad Peruana Unión.

## 🔧 Tecnologías Utilizadas

- Python 3.11+
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn

---

**Desarrollado con TensorFlow/Keras y Streamlit**
