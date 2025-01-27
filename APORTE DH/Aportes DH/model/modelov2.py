import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import folium
import joblib

# --- Cargar los datos ---
data_path = 'ruta_datos.csv'  # Cambia esto por la ubicación de tu archivo
data = pd.read_csv(data_path)

# --- Añadir Nuevas Variables ---
# Costos de transporte: Distancia al puerto más cercano
puertos = {'Puerto_1': (10.4, -75.5), 'Puerto_2': (3.9, -77.1)}  # Coordenadas de ejemplo
data['Distancia al Puerto'] = data.apply(
    lambda row: min([geodesic((row['LATITUD'], row['LONGITUD']), coord).km for coord in puertos.values()]),
    axis=1
)

# Demanda externa (valores ficticios por ahora)
productos_demanda = {'Café': 0.9, 'Caña de azúcar': 0.8, 'Frutas': 0.7, 'Maíz': 0.6, 'Plátano': 0.5, 'Arroz': 0.4}
data['Demanda Externa'] = data['Producto'].map(productos_demanda)

# Calidad de productos: Certificaciones (valores ficticios)
data['Calidad Producto'] = np.random.choice([0, 1], size=len(data))  # 0: Sin certificación, 1: Certificado

# Infraestructura: Índice ficticio
data['Indice de Infraestructura'] = np.random.uniform(0, 1, len(data))

# --- Crear Variable Objetivo ---
umbral_produccion = 2237  # Umbral de producción
umbral_crecimiento = 33.64  # Umbral de crecimiento

data['Potencial de exportación'] = (
    (data['Producción (t)'] >= umbral_produccion) &
    (data['Crecimiento'] >= umbral_crecimiento)
).astype(int)

# --- Preparar los datos para el modelo ---
required_columns = [
    'LATITUD', 'LONGITUD', 'Producción (t)', 'Crecimiento', 'Distancia al Puerto',
    'Demanda Externa', 'Calidad Producto', 'Indice de Infraestructura'
]
X = data[required_columns]
y = data['Potencial de exportación']

# --- Dividir los datos en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entrenar el Modelo ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluar el Modelo ---
y_pred = model.predict(X_test)
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# --- Guardar el Modelo ---
joblib.dump(model, 'modelo_exportacion_actualizado.pkl')
print("Modelo guardado como 'modelo_exportacion_actualizado.pkl'")

# --- Análisis de Sensibilidad ---
nuevos_umbral_produccion = 3000
nuevos_umbral_crecimiento = 40
data['Nuevo Potencial de Exportación'] = (
    (data['Producción (t)'] >= nuevos_umbral_produccion) &
    (data['Crecimiento'] >= nuevos_umbral_crecimiento)
).astype(int)

# --- Visualizaciones ---
# a) Histograma de Producción
plt.hist(data['Producción (t)'], bins=20, edgecolor='black')
plt.title('Distribución de Producción')
plt.xlabel('Producción (t)')
plt.ylabel('Frecuencia')
plt.show()

# b) Diagrama de dispersión
plt.scatter(data['Producción (t)'], data['Crecimiento'], c=data['Potencial de exportación'], cmap='coolwarm')
plt.title('Producción vs. Crecimiento')
plt.xlabel('Producción (t)')
plt.ylabel('Crecimiento (%)')
plt.colorbar(label='Potencial de exportación')
plt.show()

# c) Mapa interactivo
mapa = folium.Map(location=[4.6, -74.1], zoom_start=6)  # Coordenadas centrales de ejemplo
for _, row in data.iterrows():
    folium.Marker(
        location=[row['LATITUD'], row['LONGITUD']],
        popup=f"Municipio: {row['nombre MUNICIPIO']}<br>Producto: {row['Producto']}<br>Producción: {row['Producción (t)']}<br>Potencial: {row['Potencial de exportación']}"
    ).add_to(mapa)
mapa.save('mapa_municipios.html')
print("Mapa interactivo guardado como 'mapa_municipios.html'")
