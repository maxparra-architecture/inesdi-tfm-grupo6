import pandas as pd
from sklearn.impute import SimpleImputer

# Cargar el archivo CSV
file_path = "fuentes_combinadas.csv"  # Cambia esto por la ruta de tu archivo
data = pd.read_csv(file_path)

# --- Paso 1: Inspección Inicial ---
print("Vista previa de los datos:")
print(data.head())

# --- Paso 2: Conversión de columnas numéricas ---
print("Convirtiendo columnas numéricas...")

# Lista de columnas que deben ser numéricas
columnas_a_convertir = [
    '% pobl. con pregrado municipio', 'Valor agregado 2022',
    'Peso relativo municipal en el valor agregado departamental (%)',
    'Producción (t)', 'Crecimiento 2022'
]

# Convertir columnas numéricas (eliminando comas, % y caracteres no numéricos)
for col in columnas_a_convertir:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')

# --- Paso 3: Manejo de Valores Faltantes ---
print("Revisando valores faltantes...")
print(data.isna().sum())

# Rellenar valores faltantes con la mediana de cada columna
data[columnas_a_convertir] = data[columnas_a_convertir].fillna(data[columnas_a_convertir].median())



# --- Manejo de valores faltantes ---
# 1. Manejo de columnas numéricas
num_cols = ['Producción (t)', 'Crecimiento 2022', 'Valor agregado 2022']
imputer_num = SimpleImputer(strategy='median')  # Usar la mediana para valores numéricos
data[num_cols] = imputer_num.fit_transform(data[num_cols])

# 3. Eliminar columnas irrelevantes (si es necesario)
# data = data.drop(columns=['Columna_Innecesaria'])

print("Revisando valores faltantes...")
print(data.isna().sum())

# --- Paso 4: Normalización de Columnas ---
from sklearn.preprocessing import MinMaxScaler

# Normalizar las columnas numéricas entre 0 y 1
scaler = MinMaxScaler()
data[columnas_a_convertir] = scaler.fit_transform(data[columnas_a_convertir])


# Limpiar y convertir columnas a numérico
columnas_a_corregir = [
    '% pobl. con pregrado municipio', 'Valor agregado 2022',
    'Peso relativo municipal en el valor agregado departamental (%)'
]

for col in columnas_a_corregir:
    data[col] = pd.to_numeric(
        data[col]
        .astype(str)
        .str.replace(',', '')
        .str.replace('%', ''),
        errors='coerce'
    )

# Rellenar valores faltantes con la mediana
data[columnas_a_corregir] = data[columnas_a_corregir].fillna(data[columnas_a_corregir].median())

# Calcular el índice de competitividad
data['Indice Competitividad'] = (
    data['% pobl. con pregrado municipio'] * 0.5 +
    data['Valor agregado 2022'] * 0.3 +
    data['Peso relativo municipal en el valor agregado departamental (%)'] * 0.2
)






# --- Paso 5: Validación Final ---
print("Validando datos después de la limpieza y normalización...")
print(data.head())

# --- Paso 6: Guardar el archivo limpio ---
output_file_path = "fuentes_combinadas_limpias.csv"
data.to_csv(output_file_path, index=False)
print(f"Datos limpios guardados en: {output_file_path}")

