import pandas as pd
import joblib

# Ruta al modelo y al archivo de datos
modelo_path = 'modelo_exportacion.pkl'  # Reemplaza con la ubicación correcta
datos_path = 'municipios_potencial_exportacion.csv'  # Reemplaza con la ubicación correcta

# Cargar el modelo entrenado
print("Cargando el modelo...")
modelo = joblib.load(modelo_path)

# Definir las columnas necesarias para el modelo
required_columns = [
    'LATITUD', 'LONGITUD', 'Producción (t)', 'Crecimiento 2022',
    '% Act. primarias municipio', '% Act. secundarias municipio',
    '% Act. terciarias municipio', '% pobl. con pregrado municipio',
    'Valor agregado 2022', 'Peso relativo municipal en el valor agregado departamental (%)'
]

# Cargar los datos
print("Cargando los datos...")
datos = pd.read_csv(datos_path)

# Corregir formatos (reemplazar comas por puntos y convertir a float)
print("Corrigiendo formatos de los datos...")
for column in required_columns:
    if column in datos.columns:  # Verificar si la columna está presente en los datos
        if datos[column].dtype == 'object':  # Si la columna es texto
            datos[column] = datos[column].str.replace(',', '.').astype(float, errors='ignore')


# Validar que las columnas necesarias están presentes
missing_columns = [col for col in required_columns if col not in datos.columns]
if missing_columns:
    raise ValueError(f"Faltan las siguientes columnas en los datos: {missing_columns}")

# Calcular las columnas faltantes basadas en las transformaciones del entrenamiento
# Reconstruir columnas faltantes
datos['Primaria Dominante'] = (datos['% Act. primarias municipio'] > datos['% Act. secundarias municipio']) & \
                              (datos['% Act. primarias municipio'] > datos['% Act. terciarias municipio'])

datos['Secundaria Dominante'] = (datos['% Act. secundarias municipio'] > datos['% Act. primarias municipio']) & \
                                (datos['% Act. secundarias municipio'] > datos['% Act. terciarias municipio'])

datos['Terciaria Dominante'] = (datos['% Act. terciarias municipio'] > datos['% Act. primarias municipio']) & \
                               (datos['% Act. terciarias municipio'] > datos['% Act. secundarias municipio'])

datos['Indice Competitividad'] = (
    datos['% pobl. con pregrado municipio'] * 0.5 +
    datos['Valor agregado 2022'] * 0.3 +
    datos['Peso relativo municipal en el valor agregado departamental (%)'] * 0.2
)




# Realizar las predicciones
print("Realizando las predicciones...")
datos['Potencial de exportación'] = modelo.predict(datos[required_columns])

# Filtrar los municipios y productos con potencial de exportación
municipios_potenciales = datos[datos['Potencial de exportación'] == 1]

# Guardar los resultados en un archivo CSV
output_path = 'prediciones_modelo_abel.csv'
municipios_potenciales.to_csv(output_path, index=False)

print(f"Predicciones completadas. Los resultados se han guardado en: {output_path}")

