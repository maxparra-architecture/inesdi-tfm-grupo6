import pandas as pd
import numpy as np

# Parámetros
num_municipios = 1000
rango_produccion = (100, 10000)
rango_crecimiento = (0, 50)
rango_porcentajes = (0, 100)
umbral_produccion = 2237
umbral_crecimiento = 33.64
productos_posibles = ['café', 'yuca', 'caña de azúcar', 'tomate', 'papa', 'arroz', 'frutas tropicales', 'otros']
# Generar datos aleatorios
municipios = np.random.choice(['Municipio ' + str(i) for i in range(num_municipios)], num_municipios)
latitudes = np.random.uniform(-90, 90, num_municipios)
longitudes = np.random.uniform(-180, 180, num_municipios)
producciones = np.random.randint(rango_produccion[0], rango_produccion[1], num_municipios)
crecimientos = np.random.uniform(rango_crecimiento[0], rango_crecimiento[1], num_municipios)
act_primarias = np.random.randint(rango_porcentajes[0], rango_porcentajes[1], num_municipios)
act_secundarias = np.random.randint(rango_porcentajes[0], rango_porcentajes[1], num_municipios)
act_terciarias = 100 - act_primarias - act_secundarias
pregrado = np.random.randint(0, 100, num_municipios)
valor_agregado = np.random.randint(100000, 1000000, num_municipios)
municipio_pregrado =  np.random.randint(rango_porcentajes[0], rango_porcentajes[1], num_municipios)
peso = np.random.randint(rango_porcentajes[0], rango_porcentajes[1], num_municipios)
producto =  np.random.choice(productos_posibles, num_municipios)
# Crear DataFrame
df = pd.DataFrame({
    'MUNICIPIO': municipios,
    'LATITUD': latitudes,
    'LONGITUD': longitudes,
    'Producción (t)': producciones,
    'Crecimiento 2022': crecimientos,
    '% Act. primarias municipio': act_primarias,
    '% Act. secundarias municipio': act_secundarias,
    '% Act. terciarias municipio': act_terciarias,
    'pobl. con pregrado municipio': pregrado,
    'Valor agregado 2022': valor_agregado,
    '% pobl. con pregrado municipio' : municipio_pregrado,
    'Peso relativo municipal en el valor agregado departamental (%)' : peso,
    'Producto' : producto
})

# Exportar a CSV
df.to_csv('municipios_potencial_exportacion.csv', index=False)