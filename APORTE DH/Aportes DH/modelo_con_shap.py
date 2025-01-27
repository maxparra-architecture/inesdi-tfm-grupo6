import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import shap  # Importar SHAP para explicabilidad
import matplotlib.pyplot as plt

# 1. Cargar los datos
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
umbral_produccion = 2237
umbral_crecimiento = 33.64
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
joblib.dump(model, 'modelo_exportacion_shap.pkl')
print("Modelo guardado como 'modelo_exportacion_shap.pkl'")

# --- Aplicar SHAP para explicabilidad ---
print("Aplicando SHAP para explicabilidad...")

# Crear el objeto SHAP explainer
explainer = shap.TreeExplainer(model)

# Calcular valores SHAP para el conjunto de prueba
shap_values = explainer.shap_values(X_test)

# Visualización: Importancia global de las características
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# Visualización: Razones específicas para una predicción
# Cambia el índice para analizar diferentes municipios
municipio_index = 0
shap.force_plot(
    explainer.expected_value[1], 
    shap_values[1][municipio_index], 
    X_test.iloc[municipio_index]
)

# Guardar explicaciones para todas las predicciones
X_test['SHAP_Explicaciones'] = [
    ", ".join(f"{feature}: {shap_value:.2f}" for feature, shap_value in zip(X_test.columns, shap_values[1][i]))
    for i in range(len(X_test))
]
X_test.to_csv('predicciones_con_explicaciones.csv', index=False)
print("Explicaciones guardadas en 'predicciones_con_explicaciones.csv'")
