import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Cargar los datos
# Reemplaza 'ruta_datos.csv' con la ubicación de tu archivo CSV
data = pd.read_csv('fuentes_combinadas.csv')

# 2. Definir las columnas necesarias y la variable objetivo
required_columns = [
    'LATITUD', 'LONGITUD', 'Producción (t)', 'Crecimiento 2022',
    '% Act. primarias municipio', '% Act. secundarias municipio',
    '% Act. terciarias municipio', '% pobl. con pregrado municipio',
    'Valor agregado 2022', 'Peso relativo municipal en el valor agregado departamental (%)'
]

# Verificar si las columnas están presentes
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Faltan las siguientes columnas en los datos: {missing_columns}")

# Crear la variable objetivo "Potencial de exportación"
umbral_produccion = 2237  # Cuartil superior para la producción (ejemplo)
umbral_crecimiento = 33.64  # Cuartil superior para el crecimiento (ejemplo)

data['Potencial de exportación'] = (
    (data['Producción (t)'] >= umbral_produccion) &
    (data['Crecimiento 2022'] >= umbral_crecimiento)
).astype(int)

# 3. Separar las variables predictoras (X) y la variable objetivo (y)
X = data[required_columns]
y = data['Potencial de exportación']

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Crear y entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar el modelo en los datos de prueba
y_pred = model.predict(X_test)
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 7. Guardar el modelo entrenado
# Reemplaza 'modelo_exportacion.pkl' con la ubicación deseada
joblib.dump(model, 'modelo_exportacion.pkl')
print("Modelo guardado como 'modelo_exportacion.pkl'")
