"""
Script para entrenar una Red Neuronal Artificial (RNA)
para predecir el rendimiento académico de estudiantes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import json

# Cargar datos
print("Cargando datos...")
df = pd.read_csv("estudiantes_data (1).csv")
print(f"Datos cargados: {df.shape}")

# Separar características (X) y objetivo (y)
# Vamos a predecir la Nota_promedio
X = df.drop('Nota_promedio', axis=1)
y = df['Nota_promedio']

# Dividir en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Entrenamiento: {X_train.shape[0]} registros")
print(f"Prueba: {X_test.shape[0]} registros")

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de RNA
print("\nCreando modelo de Red Neuronal...")
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Salida para regresión
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

# Callback para detener entrenamiento si no mejora
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Entrenar el modelo
print("\nEntrenando modelo...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluar el modelo
print("\nEvaluando modelo...")
train_loss, train_mae, train_mse = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nResultados:")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Guardar el modelo y el scaler
print("\nGuardando modelo y scaler...")
model.save("modelo_estudiantes.h5")
joblib.dump(scaler, "scaler_estudiantes.pkl")

# Guardar nombres de características
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Guardar metadata
metadata = {
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_mse': float(train_mse),
    'test_mse': float(test_mse),
    'feature_names': feature_names,
    'n_features': len(feature_names)
}

with open("metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Modelo entrenado y guardado exitosamente!")
print(f"   - modelo_estudiantes.h5")
print(f"   - scaler_estudiantes.pkl")
print(f"   - feature_names.pkl")
print(f"   - metadata.json")
