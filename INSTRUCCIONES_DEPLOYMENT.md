# 🚀 Instrucciones para Desplegar en Streamlit Cloud

## Opción 1: Despliegue en Streamlit Cloud (Gratis)

### Paso 1: Preparar el repositorio

1. Asegúrate de que todos los archivos necesarios estén en tu repositorio:
   - `app_streamlit_estudiantes.py`
   - `modelo_estudiantes.h5`
   - `scaler_estudiantes.pkl`
   - `feature_names.pkl`
   - `metadata.json`
   - `requirements.txt`

### Paso 2: Subir a GitHub

```bash
cd /home/user/project-analisize
git add EsMio/
git commit -m "Agregar aplicación RNA de rendimiento académico"
git push origin claude/streamlit-neural-network-app-014My5cXefn9DFb6Ypu7JW1Q
```

### Paso 3: Desplegar en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Click en "New app"
4. Selecciona:
   - **Repository:** tu repositorio
   - **Branch:** la rama actual
   - **Main file path:** `EsMio/app_streamlit_estudiantes.py`
5. Click en "Deploy"

¡Listo! Tu aplicación estará disponible en unos minutos.

## Opción 2: Ejecutar Localmente

### Instalación

```bash
cd EsMio
pip install -r requirements.txt
```

### Ejecutar

```bash
streamlit run app_streamlit_estudiantes.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📝 Notas Importantes

- Los archivos `.h5`, `.pkl` y `.json` **deben estar** en el mismo directorio que `app_streamlit_estudiantes.py`
- Si ejecutas la app desde otra ubicación, ajusta las rutas en el código
- El modelo ya está entrenado, no necesitas ejecutar `train_model.py` de nuevo a menos que quieras reentrenar

## 🔧 Solución de Problemas

### Error: "No module named 'tensorflow'"

```bash
pip install tensorflow
```

### Error: "No se encontró el modelo"

Asegúrate de estar en el directorio `EsMio` cuando ejecutes la aplicación, o ajusta las rutas en el código:

```python
MODEL_PATH = "EsMio/modelo_estudiantes.h5"
SCALER_PATH = "EsMio/scaler_estudiantes.pkl"
# etc...
```

### La app se carga muy lento en Streamlit Cloud

Esto es normal en el primer despliegue. TensorFlow es pesado y puede tardar 2-3 minutos en cargar.

## 📊 Rendimiento del Modelo

- **MAE (Error Absoluto Medio) en prueba:** 1.07
- **MSE (Error Cuadrático Medio) en prueba:** 1.76
- **Precisión:** El modelo predice con ~1 punto de error en promedio

## 🎯 Características

- 14 características de entrada
- Red neuronal con 4 capas (64-32-16-1 neuronas)
- Dropout para evitar sobreajuste
- Early stopping implementado
- Interfaz web interactiva con Streamlit
