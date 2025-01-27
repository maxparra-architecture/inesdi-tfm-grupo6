import pandas as pd

# Ruta del archivo CSV
input_file_path = "fuentes_combinadas.csv"  # Cambia esto por la ruta de tu archivo
output_file_path = "archivo_convertido.csv"

# Cargar el archivo CSV
print("Cargando el archivo...")
data = pd.read_csv(input_file_path)

# Listar las columnas del archivo
print("Columnas en el archivo:")
print(data.columns)

# Identificar columnas numéricas que puedan estar en formato string
columns_to_convert = data.select_dtypes(include=['object']).columns

print("\nColumnas a convertir a float:")
print(columns_to_convert)

# Convertir columnas con datos string a float
for col in columns_to_convert:
    try:
        # Reemplazar caracteres problemáticos y convertir
        data[col] = pd.to_numeric(
            data[col].astype(str).str.replace(',', '').str.replace('%', '').str.replace(' ', ''), 
            errors='coerce'
        )
    except Exception as e:
        print(f"Error al convertir la columna {col}: {e}")

# Verificar valores faltantes después de la conversión
print("\nValores faltantes después de la conversión:")
print(data.isna().sum())

# Rellenar valores faltantes con la mediana
data = data.fillna(data.median())

# Guardar el archivo convertido
data.to_csv(output_file_path, index=False)
print(f"\nArchivo convertido guardado en: {output_file_path}")
