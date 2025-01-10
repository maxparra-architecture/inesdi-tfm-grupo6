
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Cargar el modelo entrenado
model = joblib.load('/mnt/data/modelo_exportacion.pkl')

# Función para predecir potencial de exportación
def predecir_potencial(datos_nuevos):
    # Convertir a DataFrame si los datos no lo son
    if not isinstance(datos_nuevos, pd.DataFrame):
        datos_nuevos = pd.DataFrame(datos_nuevos)
    
    # Asegurar que los datos tienen las columnas necesarias
    required_columns = [
        'LATITUD', 'LONGITUD', 'Producción (t)', 'Crecimiento 2022',
        '% Act. primarias municipio', '% Act. secundarias municipio', 
        '% Act. terciarias municipio', '% pobl. con pregrado municipio', 
        'Valor agregado 2022', 'Peso relativo municipal en el valor agregado departamental (%)'
    ]
    if not all(col in datos_nuevos.columns for col in required_columns):
        raise ValueError(f"Los datos deben contener las siguientes columnas: {required_columns}")

    # Predecir
    predicciones = model.predict(datos_nuevos[required_columns])
    return predicciones

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    datos_ejemplo = {
        'LATITUD': [6.25],
        'LONGITUD': [-75.56],
        'Producción (t)': [5000],
        'Crecimiento 2022': [10],
        '% Act. primarias municipio': [30],
        '% Act. secundarias municipio': [20],
        '% Act. terciarias municipio': [50],
        '% pobl. con pregrado municipio': [25],
        'Valor agregado 2022': [10000],
        'Peso relativo municipal en el valor agregado departamental (%)': [15]
    }
    datos_df = pd.DataFrame(datos_ejemplo)
    print(predecir_potencial(datos_df))
