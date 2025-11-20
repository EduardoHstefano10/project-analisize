"""
Script para entrenar Red Neuronal para Predicción de Deserción Académica
Siguiendo exactamente las especificaciones del PDF del proyecto
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("ENTRENAMIENTO DE RED NEURONAL PARA PREDICCIÓN DE DESERCIÓN ACADÉMICA")
print("="*70)

# ========== CARGA DE DATOS ==========
print("\n[1] Cargando datos...")
df = pd.read_csv("datos_desercion_academica.csv")
print(f"✓ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")

# ========== PREPROCESAMIENTO ==========
print("\n[2] Preprocesamiento de datos...")

# Separar características y variable objetivo
X = df.drop('Riesgo_deserción', axis=1)
y = df['Riesgo_deserción']

print(f"✓ Características (X): {X.shape}")
print(f"✓ Variable objetivo (y): {y.shape}")

# Aplicar One-Hot Encoding a las variables categóricas
print("\n[3] Aplicando One-Hot Encoding a variables categóricas...")
X_encoded = pd.get_dummies(X, drop_first=False)
print(f"✓ Características después de encoding: {X_encoded.shape}")

# Aplicar LabelEncoder a la variable objetivo
print("\n[4] Codificando variable objetivo con LabelEncoder...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"✓ Clases encontradas: {le.classes_}")
print(f"✓ Número de clases: {len(le.classes_)}")

# Convertir a categorical (one-hot encoding para la salida)
y_categorical = to_categorical(y_encoded)
print(f"✓ Variable objetivo convertida a categorical: {y_categorical.shape}")

# División 80/20 como indica el PDF
print("\n[5] Dividiendo datos: 80% entrenamiento, 20% prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"✓ Conjunto de entrenamiento: {X_train.shape[0]} registros")
print(f"✓ Conjunto de prueba: {X_test.shape[0]} registros")

# Escalado de datos
print("\n[6] Escalando datos con StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Datos escalados correctamente")

# ========== CONSTRUCCIÓN DEL MODELO ==========
print("\n[7] Construyendo arquitectura de Red Neuronal...")
print("    Según especificaciones del PDF:")
print("    - Capa entrada: 128 neuronas, activación ReLU")
print("    - Dropout")
print("    - Capa oculta: 64 neuronas, activación ReLU")
print("    - Capa salida: 5 neuronas, activación Softmax")

model = Sequential([
    # Capa de entrada: 128 neuronas, activación ReLU
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),

    # Dropout para regularización
    Dropout(0.3),

    # Capa oculta: 64 neuronas, activación ReLU
    Dense(64, activation='relu'),

    # Capa de salida: 5 neuronas (5 clases), activación Softmax
    Dense(len(le.classes_), activation='softmax')
])

print("✓ Modelo construido exitosamente")

# ========== COMPILACIÓN ==========
print("\n[8] Compilando modelo...")
print("    - Optimizador: Adam")
print("    - Función de pérdida: categorical_crossentropy")
print("    - Métricas: accuracy")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Modelo compilado exitosamente")

# Mostrar resumen del modelo
print("\n[9] Resumen de la arquitectura:")
model.summary()

# ========== ENTRENAMIENTO ==========
print("\n[10] Entrenando modelo...")
print("     - Épocas: 40")
print("     - Batch size: 32")
print("     - Validación: 20%")

history = model.fit(
    X_train_scaled, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\n✓ Entrenamiento completado")

# ========== EVALUACIÓN ==========
print("\n[11] Evaluando modelo en conjunto de prueba...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"✓ Pérdida en prueba: {test_loss:.4f}")
print(f"✓ Precisión en prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Predicciones
print("\n[12] Generando reporte de clasificación...")
y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Reporte de clasificación
print("\n" + "="*70)
print("REPORTE DE CLASIFICACIÓN")
print("="*70)
print(classification_report(
    y_test_classes,
    y_pred_classes,
    target_names=le.classes_,
    digits=4
))

# Matriz de confusión
print("\n[13] Generando matriz de confusión...")
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Matriz de Confusión - Riesgo de Deserción')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
print("✓ Matriz de confusión guardada en: matriz_confusion.png")

# Gráficas de entrenamiento
print("\n[14] Generando gráficas de entrenamiento...")

# Gráfica de precisión
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Evolución de la Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Evolución de la Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evolucion_modelo.png', dpi=300, bbox_inches='tight')
print("✓ Gráficas guardadas en: evolucion_modelo.png")

# ========== GUARDADO DEL MODELO ==========
print("\n[15] Guardando modelo y objetos...")

# Guardar modelo en formato .h5 (como indica el PDF)
model.save("modelo_riesgo_desercion.h5")
print("✓ Modelo guardado: modelo_riesgo_desercion.h5")

# También guardar en formato .keras (formato moderno)
model.save("modelo_riesgo_desercion.keras")
print("✓ Modelo guardado: modelo_riesgo_desercion.keras")

# Guardar LabelEncoder
joblib.dump(le, "label_encoder.pkl")
print("✓ LabelEncoder guardado: label_encoder.pkl")

# Guardar nombres de columnas (feature names)
joblib.dump(X_encoded.columns.tolist(), "columnas_X.pkl")
print("✓ Nombres de características guardados: columnas_X.pkl")

# Guardar scaler
joblib.dump(scaler, "scaler_estudiantes.pkl")
print("✓ Scaler guardado: scaler_estudiantes.pkl")

# ========== RESUMEN FINAL ==========
print("\n" + "="*70)
print("RESUMEN FINAL")
print("="*70)
print(f"Registros totales:        {len(df)}")
print(f"Características (encoded): {X_encoded.shape[1]}")
print(f"Clases:                   {len(le.classes_)}")
print(f"Precisión en prueba:      {test_accuracy*100:.2f}%")
print(f"Épocas entrenadas:        40")
print(f"Arquitectura:             128 -> Dropout -> 64 -> 5 (Softmax)")
print("\nArchivos generados:")
print("  ✓ modelo_riesgo_desercion.h5")
print("  ✓ modelo_riesgo_desercion.keras")
print("  ✓ label_encoder.pkl")
print("  ✓ columnas_X.pkl")
print("  ✓ scaler_estudiantes.pkl")
print("  ✓ matriz_confusion.png")
print("  ✓ evolucion_modelo.png")
print("="*70)
print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*70)
