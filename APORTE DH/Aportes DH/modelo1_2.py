import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# --- Función principal ---
def entrenar_y_guardar_modelo_mejorado(data_path, output_model_path):
    """
    Entrena un modelo de Random Forest para predecir el potencial de exportación mejorado con nuevas funcionalidades.

    :param data_path: Ruta al archivo CSV que contiene los datos reales.
    :param output_model_path: Ruta donde se guardará el modelo entrenado.
    """
    # --- Cargar los datos ---
    print("Cargando los datos...")
    data = pd.read_csv(data_path)

    # --- Validar columnas requeridas ---
    required_columns = [
        'LATITUD', 'LONGITUD', 'Producción (t)', 'Crecimiento 2022',
        '% Act. primarias municipio', '% Act. secundarias municipio',
        '% Act. terciarias municipio', '% pobl. con pregrado municipio',
        'Valor agregado 2022', 'Peso relativo municipal en el valor agregado departamental (%)'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Faltan las siguientes columnas en los datos: {missing_columns}")

    # --- Transformaciones Adicionales ---
    print("Realizando transformaciones adicionales...")
    # 1. Crear indicadores económicos basados en actividades
    data['Primaria Dominante'] = (data['% Act. primarias municipio'] > data['% Act. secundarias municipio']) & \
                                 (data['% Act. primarias municipio'] > data['% Act. terciarias municipio'])
    data['Secundaria Dominante'] = (data['% Act. secundarias municipio'] > data['% Act. primarias municipio']) & \
                                   (data['% Act. secundarias municipio'] > data['% Act. terciarias municipio'])
    data['Terciaria Dominante'] = (data['% Act. terciarias municipio'] > data['% Act. primarias municipio']) & \
                                   (data['% Act. terciarias municipio'] > data['% Act. secundarias municipio'])

    # 2. Calcular un índice combinado de competitividad basado en educación y valor agregado
    data['Indice Competitividad'] = (
        data['% pobl. con pregrado municipio'] * 0.5 +  # Ponderación del nivel educativo
        data['Valor agregado 2022'] * 0.3 +            # Ponderación del valor agregado
        data['Peso relativo municipal en el valor agregado departamental (%)'] * 0.2  # Ponderación del peso relativo
    )

    # --- Crear la variable objetivo ---
    print("Creando variable objetivo...")
    umbral_produccion = data['Producción (t)'].quantile(0.75)  # Cuartil superior para producción
    umbral_crecimiento = data['Crecimiento 2022'].quantile(0.75)  # Cuartil superior para crecimiento
    data['Potencial de exportación'] = (
        (data['Producción (t)'] >= umbral_produccion) &
        (data['Crecimiento 2022'] >= umbral_crecimiento)
    ).astype(int)

    # --- Preparar los datos ---
    all_columns = required_columns + ['Primaria Dominante', 'Secundaria Dominante', 'Terciaria Dominante', 'Indice Competitividad']
    X = data[all_columns]
    y = data['Potencial de exportación']

    # Imputar valores faltantes en las columnas necesarias
    imputer = SimpleImputer(strategy='mean')
    data[required_columns] = imputer.fit_transform(data[required_columns])

    # --- Dividir los datos en entrenamiento y prueba ---
    print("Dividiendo los datos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Entrenar el modelo ---
    print("Entrenando el modelo...")


    # Crear y entrenar el modelo
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)


    # --- Evaluar el modelo ---
    print("Evaluando el modelo...")
    y_pred = model.predict(X_test)
    print("Precisión del modelo:", accuracy_score(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # --- Guardar el modelo ---
    joblib.dump(model, output_model_path)
    print(f"Modelo guardado como '{output_model_path}'")


# --- Uso del script ---
if __name__ == "__main__":
    # Rutas de entrada y salida
    ruta_datos = "./fuentes_combinadas_limpias.csv"  # Cambia por la ubicación del archivo de datos reales
    ruta_modelo_salida = "modelo_exportacion_mejorado.pkl"  # Ruta para guardar el modelo entrenado

    # Entrenar y guardar el modelo
    entrenar_y_guardar_modelo_mejorado(ruta_datos, ruta_modelo_salida)
