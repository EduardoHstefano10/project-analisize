# 🎓 Predictor de Rendimiento Académico con RNA

Aplicación simple de Streamlit que utiliza Redes Neuronales Artificiales (RNA) para predecir el rendimiento académico de estudiantes.

## 📋 Descripción

Esta aplicación utiliza TensorFlow/Keras para entrenar una red neuronal que predice la nota promedio de un estudiante basándose en 14 características académicas y personales.

## 🚀 Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📊 Uso

### Paso 1: Entrenar el modelo

Antes de usar la aplicación, debes entrenar el modelo de RNA:

```bash
cd EsMio
python train_model.py
```

Esto generará:
- `modelo_estudiantes.h5` - Modelo entrenado
- `scaler_estudiantes.pkl` - Escalador de datos
- `feature_names.pkl` - Nombres de características
- `metadata.json` - Metadatos del modelo

### Paso 2: Ejecutar la aplicación Streamlit

```bash
streamlit run app_streamlit_estudiantes.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## 📈 Características del Modelo

El modelo utiliza las siguientes características para predecir la nota promedio:

1. **Promedio Ponderado**
2. **Créditos Matriculados**
3. **Porcentaje Créditos Aprobados**
4. **Cursos Desaprobados**
5. **Asistencia**
6. **Retiros de Cursos**
7. **Edad**
8. **Horas de Trabajo por Semana**
9. **Año de Ingreso**
10. **Número de Ciclos Académicos**
11. **Cursos Matriculados por Ciclo**
12. **Horas de Estudio por Semana**
13. **Índice de Regularidad**
14. **Intentos de Aprobación de Curso**

## 🧠 Arquitectura de la Red Neuronal

- **Capa de entrada:** 14 características
- **Capa oculta 1:** 64 neuronas + Dropout (30%)
- **Capa oculta 2:** 32 neuronas + Dropout (20%)
- **Capa oculta 3:** 16 neuronas
- **Capa de salida:** 1 neurona (regresión)

## 📁 Archivos

- `estudiantes_data (1).csv` - Dataset original
- `train_model.py` - Script de entrenamiento
- `app_streamlit_estudiantes.py` - Aplicación Streamlit
- `requirements.txt` - Dependencias del proyecto

## 🔧 Tecnologías

- **TensorFlow/Keras** - Framework de deep learning
- **Streamlit** - Framework para la aplicación web
- **Scikit-learn** - Preprocesamiento de datos
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas

## 📝 Notas

- El modelo usa MAE (Error Absoluto Medio) como métrica principal
- Se implementa Early Stopping para evitar sobreajuste
- Los datos se normalizan usando StandardScaler
- División de datos: 80% entrenamiento, 20% prueba
