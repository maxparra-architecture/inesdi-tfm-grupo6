import zipfile
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import plotly.graph_objs as go
import folium
from streamlit_folium import st_folium
import os
#import locale
from babel.numbers import format_number
from babel.numbers import format_decimal
from menu import menu

# def format_deci(number):
#     # Configurar la localización para español
#     locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
#     # Función para formatear los números
#     formatted_number = locale.format_string("%0.1f", number, grouping=True)
#     return formatted_number

# def format_int(number):
#     # Configurar la localización para español
#     locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
#     # Función para formatear los números
#     formatted_number = locale.format_string("%.0f", number, grouping=True)
#     return formatted_number

def format_int(number):
    return format_number(number, locale='es_ES')

def format_deci(number):
    return format_decimal(number, format='#,##0.0', locale='es_ES')

# Aumentar número de columnas que se pueden ver
pd.options.display.max_columns = None
# En los dataframes, mostrar los float con dos decimales
pd.options.display.float_format = '{:,.2f}'.format
# Cada columna será tan grande como sea necesario para mostrar todo su contenido
pd.set_option('display.max_colwidth', 0)

# Rutas
path_cwd = os.getcwd()
path_data = path_cwd + "/data/"
path_img  = path_cwd + "/img/"

# Archivos
a_base = "Tejido_Municipios.txt"
#a_base = "Base.zip"
a_eva = "eva.txt"
a_va = "Valor_Agregado.txt"
a_pdet = "PDET_y_ZOMAC.txt"
a_ipm = "IPM.txt"
a_pob = "Proyecciones_poblacion.txt"
a_censo = "Datos_censo_por_municipios.txt"
a_ubic = "DIVIPOLA_Municipios.xlsx"
a_graduados = "Graduados_municipio_2018-2022.csv"
a_tec = "tec.txt"


# Configuración página web
st.set_page_config(page_title="Perfiles Municipios de Colombia", page_icon = '🌎', layout="wide",  
                   initial_sidebar_state="expanded") 

# Funciones para cargar datos con caché
@st.cache_data
def cargar_general(path):
    return pd.read_csv(path, decimal=',', sep='|', encoding='utf-8', converters={'Cod. Municipio': str})

@st.cache_data
def cargar_base(path):
    return pd.read_csv(path, sep="|", decimal=",", encoding='utf-8', converters={'Cod. Depto': str, 'Cod. Municipio': str, 'CIIU Rev 4 principal': str})

#@st.cache_data
#def cargar_base(zip_path, file_name):
#    with zipfile.ZipFile(zip_path, 'r') as z:
#        with z.open(file_name) as f:
#            df = pd.read_csv(f, sep="|", decimal=",", encoding='utf-8', converters={'Cod. Depto': str, 'Cod. Municipio': str, 'CIIU Rev 4 principal': str})
#            df = df.drop(columns=["Tamaño empresa RUES"])
#            df = df.rename(columns={"Tamaño empresa Supersociedades": "Tamaño", "Cadena CIIU principal": "Cadena productiva"})
#            return df
        
@st.cache_data
def cargar_ubicacion(path):
    df = pd.read_excel(path, skiprows=10, converters={'Código .1': str})
    df = df[['Código .1', 'Nombre', 'Nombre.1', 'LATITUD', 'LONGITUD']]
    df = df[df['Código .1'].notna()]
    df.rename(columns={'Código .1': 'Cod. Municipio', 'Nombre': 'Departamento', 'Nombre.1': 'Municipio'}, inplace=True)
    return df

@st.cache_data
def cargar_eva(path):
    return pd.read_csv(path, decimal=',', sep='|', encoding='utf-8', converters={'Código Dane municipio': str})

@st.cache_data
def cargar_graduados(path):
    return pd.read_csv(path, sep=';', encoding='utf-8', converters={'CÓDIGO DEL MUNICIPIO (PROGRAMA)': str})

@st.cache_data
def cargar_tec(path):
    return pd.read_csv(path, decimal=',', sep='|', encoding='utf-8', converters={'CODIGO': str})

# Cargar bases de datos
valor_agregado = cargar_general(path_data + a_va)
pdet = cargar_general(path_data + a_pdet)
ipm = cargar_general(path_data + a_ipm)
pob2 = cargar_general(path_data + a_pob)
censo = cargar_general(path_data + a_censo)
base = cargar_base(path_data + a_base)
#base = cargar_base(path_data + a_base, "Base.txt")
ubicacion = cargar_ubicacion(path_data + a_ubic)
eva = cargar_eva(path_data + a_eva)
grad = cargar_graduados(path_data + a_graduados)
tec = cargar_tec(path_data + a_tec)

# --------------- Sidebar -------------------------------------------

# Logo ProColombia
st.sidebar.image(path_img + "Logo_ProColombia.png", use_column_width=True)
st.sidebar.divider()
st.sidebar.text("")

# Menú
st.sidebar.page_link("app.py", label="▶ :gray[Página Principal]")
st.sidebar.page_link("pages/1-Perfil.py", label="▶ :gray[Perfiles municipios]")
st.sidebar.page_link("pages/2-Metodologia.py", label="▶ :gray[Metodología]")
st.sidebar.text("")
st.sidebar.text("")

# Filtrar el municipio de interés
st.sidebar.title('Escoja el municipio de interés') 
# Lista de departamentos
depto0 = base[base['Departamento']!='No determinado']
depto = sorted(depto0['Departamento'].unique().tolist())
# Crear selectbox departamento
index1 = depto.index("Chocó")
depto_seleccionado = st.sidebar.selectbox("Seleccione el departamento", depto, index=index1)
# Lista de municipios
mpio = sorted(base[base['Departamento']==depto_seleccionado]['Municipio'].unique().tolist())
# Crear selectbox municipio
mpio_seleccionado = st.sidebar.selectbox("Seleccione el municipio", mpio)

# Filtrar la información por el territorio de interés
cod_mpio_selec = base[(base['Departamento']==depto_seleccionado)&(base['Municipio']==mpio_seleccionado)]['Cod. Municipio'].values[0]
tejido=base[base['Cod. Municipio']==cod_mpio_selec]
#datos_mun=general[general['Cod. Municipio']==cod_mpio_selec]
mapa = ubicacion[ubicacion['Cod. Municipio'] == cod_mpio_selec]
eva_mun = eva[eva['Código Dane municipio']==cod_mpio_selec]
grad_mun = grad[grad['CÓDIGO DEL MUNICIPIO (PROGRAMA)']==cod_mpio_selec]
valor_agregado_mun = valor_agregado[valor_agregado['Cod. Municipio']==cod_mpio_selec]
pdet_mun = pdet[pdet['Cod. Municipio']==cod_mpio_selec]
ipm_mun = ipm[ipm['Cod. Municipio']==cod_mpio_selec]
pob_mun = pob2[pob2['Cod. Municipio']==cod_mpio_selec]
censo_mun = censo[censo['Cod. Municipio']==cod_mpio_selec]
tec_mun = tec[tec['CODIGO']==cod_mpio_selec]

# Elaborado por
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.subheader("Elaborado por:") 
st.sidebar.markdown("####  Coordinación de Analítica, Gerencia de Inteligencia Comercial, ProColombia.")
st.sidebar.divider()

# Logo Ministerio
st.sidebar.image(path_img + "Logo_MinCit.png", use_column_width=True)

# ------------- Tablero -------------------------------

#st.title(f'Perfil territorio: {mpio_seleccionado} - {depto_seleccionado}')
st.markdown(f'<h1 style="text-align: center;">Perfil municipio: {mpio_seleccionado} - {depto_seleccionado}</h1>', unsafe_allow_html=True)

st.divider()

# Datos generales del municipio
st.header('🌐 **Ubicación geográfica**')

# Mapa del municipio
if not mapa['Cod. Municipio'].isnull().all():
    # Obtener las coordenadas del municipio seleccionado
    municipio_lat = mapa['LATITUD']
    municipio_lon = mapa['LONGITUD']
    # Obtener las coordenadas del centro de Colombia
    colombia_center = [4.5709, -74.2973]
    # Crear el mapa de Colombia
    m = folium.Map(location=colombia_center, zoom_start=5)
    # Añadir marcador para el municipio seleccionado
    folium.Marker(location=[municipio_lat, municipio_lon], popup=f'{mpio_seleccionado} - {depto_seleccionado}').add_to(m)
    # Mostrar el mapa en Streamlit ajustándose automáticamente al tamaño del contenedor
    st_folium(m, width='100%', height=500)
else:
    st.markdown('##### Para este municipio no hay información.')

st.markdown("**Fuente:** Divipola Marzo 2024, DANE.")

st.divider()

st.header('🔍 **Información general**')

if not pdet_mun['Cod. Municipio'].isnull().all():
    c1,c2,c3 = st.columns(3)
    with c1:
        # Territorios PDET
        pdet_mun['PDET'] = 'Es municipio PDET'
        pdet_mun['Subregión PDET'] = pdet_mun['Subregión PDET'].str.title()
        pdet_mun['Metrica PDET'] = np.where(pdet_mun['Subregión PDET'].notna(), pdet_mun['PDET'] + ' - Subregión ' + pdet_mun['Subregión PDET'], 'No es municipio PDET')
        pdet = pdet_mun['Metrica PDET'].values[0]
        st.markdown(f'#### {pdet}')
    with c2:
        # Territorios Zomac
        pdet_mun['PDET'] = 'PDET'
        pdet_mun['Metrica ZOMAC'] = np.where(pdet_mun['ZOMAC']==1, 'Es municipio ZOMAC', 'No es municipio ZOMAC')
        zomac = pdet_mun['Metrica ZOMAC'].values[0]
        st.markdown(f'#### {zomac}')
    with c3:
        # Población
        pob = pob_mun['Población municipio'].values[0]
        st.markdown("#### Población 2024")
        st.subheader(f'{format_int(pob)} habitantes')
        if pob < 200000:
            st.markdown("##### Menos de 200.000 habitantes")
        else:
            st.markdown("##### Más de 200.000 habitantes")
else:
    st.markdown('##### Para este municipio no hay información.') 

st.markdown("**Fuentes:** Ley 893 de 2017 - Decreto 1650 de 2017 - Proyecciones de población municipal por área, sexo y edad para 2024, DANE.") 

st.divider()

st.header('👨‍👩‍👧‍👦 **Población diferencial**')

if not pob_mun['Cod. Municipio'].isnull().all():
    c1,c2 = st.columns(2)
    with c1:
        # Sexo
        femenino = pob_mun['% mujeres municipio'].values[0]
        masculino = 100 - femenino
        femenino_formateado = format_deci(femenino)
        masculino_formateado = format_deci(masculino)
        fig_muj = go.Figure(data=[go.Pie(labels=['Femenino', 'Masculino'], 
                                    values=[femenino, masculino], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(255, 218, 0)', 'rgb(0, 109, 254)']),
                                    insidetextorientation='auto',
                                    customdata=[[femenino_formateado, femenino], [masculino_formateado, masculino]])])
        fig_muj.update_traces(textposition='outside', 
                        textinfo='none',
                        hoverinfo='none',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
        fig_muj.update_layout(showlegend=False, annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'Año 2024',
            'showarrow': False,
            'font': {'size': 20}
        }])
        st.plotly_chart(fig_muj, use_container_width=True)
    with c2:
        # Jóvenes
        joven = pob_mun['% jóvenes municipio'].values[0]
        no_joven = 100 - joven
        joven_formateado = format_deci(joven)
        no_joven_formateado = format_deci(no_joven)
        fig_jov = go.Figure(data=[go.Pie(labels=['Jóvenes', 'Resto <br> de <br> población'], 
                                    values=[joven, no_joven], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(255, 218, 0)', 'rgb(0, 109, 254)']),
                                    insidetextorientation='horizontal',
                                    customdata=[[joven_formateado, joven], [no_joven_formateado, no_joven]])])
        fig_jov.update_traces(textposition='outside', 
                        textinfo='none',
                        hoverinfo='none',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
        fig_jov.update_layout(showlegend=False, annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'Año 2024',
            'showarrow': False,
            'font': {'size': 20}
        }])
        st.plotly_chart(fig_jov, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        # Grupos étnicos
        etnico = censo_mun['% grupos étnicos municipio'].values[0]
        no_etnico = 100 - etnico
        etnico_formateado = format_deci(etnico)
        no_etnico_formateado = format_deci(no_etnico)
        fig_etn = go.Figure(data=[go.Pie(labels=['Grupos <br> étnicos', 'Resto <br> de <br> población'], 
                                    values=[etnico, no_etnico], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(255, 218, 0)', 'rgb(0, 109, 254)']),
                                    insidetextorientation='horizontal',
                                    customdata=[[etnico_formateado, etnico], [no_etnico_formateado, no_etnico]])])
        fig_etn.update_traces(textposition='outside', 
                        textinfo='none',
                        hoverinfo='none',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
        fig_etn.update_layout(showlegend=False, annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'Censo 2018',
            'showarrow': False,
            'font': {'size': 20}
        }])
        st.plotly_chart(fig_etn, use_container_width=True)
    with c2:
        # Discapacitados
        discap = censo_mun['% pobl. con discapacidad municipio'].values[0]
        no_discap = 100 - discap
        discap_formateado = format_deci(discap)
        no_discap_formateado = format_deci(no_discap)
        fig_disc = go.Figure(data=[go.Pie(labels=['Con <br> discapacidad', 'Resto <br> de <br> población'], 
                                    values=[discap, no_discap], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(0, 109, 254)', 'rgb(255, 218, 0)']),
                                    insidetextorientation='horizontal',
                                    customdata=[[discap_formateado, discap], [no_discap_formateado, no_discap]])])
        fig_disc.update_traces(textposition='outside', 
                        textinfo='none',
                        hoverinfo='none',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
        fig_disc.update_layout(showlegend=False, annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'Censo 2018',
            'showarrow': False,
            'font': {'size': 20}
        }])
        st.plotly_chart(fig_disc, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        # Pobreza
        pobre = ipm_mun['% pobreza municipio'].values[0]
        no_pobre = 100 - pobre
        pobre_formateado = format_deci(pobre)
        no_pobre_formateado = format_deci(no_pobre)
        fig_pobre = go.Figure(data=[go.Pie(labels=['En <br> situación <br> de <br> pobreza', 'Resto <br> de <br> población'], 
                                    values=[pobre, no_pobre], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(255, 218, 0)', 'rgb(0, 109, 254)']),
                                    insidetextorientation='horizontal',
                                    customdata=[[pobre_formateado, pobre], [no_pobre_formateado, no_pobre]])])
        fig_pobre.update_traces(textposition='outside', 
                        textinfo='none',
                        hoverinfo='none',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
        fig_pobre.update_layout(showlegend=False, annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'IPM 2018',
            'showarrow': False,
            'font': {'size': 20}
        }])
        st.plotly_chart(fig_pobre, use_container_width=True)
    with c2:
        # Informalidad
        informal = ipm_mun['% informalidad municipio'].values[0]
        no_informal = 100 - informal
        informal_formateado = format_deci(informal)
        no_informal_formateado = format_deci(no_informal)
        fig_etn = go.Figure(data=[go.Pie(labels=['Hogares <br> con <br> ocupados <br> informales', 'Resto <br> de <br> hogares'], 
                                    values=[informal, no_informal], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(255, 218, 0)', 'rgb(0, 109, 254)']),
                                    insidetextorientation='horizontal',
                                    customdata=[[informal_formateado, informal], [no_informal_formateado, no_informal]])])
        fig_etn.update_traces(textposition='outside', 
                        textinfo='none',
                        hoverinfo='none',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
        fig_etn.update_layout(showlegend=False, annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'IPM 2018',
            'showarrow': False,
            'font': {'size': 20}
        }])
        st.plotly_chart(fig_etn, use_container_width=True)  
else:
    st.markdown('##### Para este municipio no hay información.') 

st.markdown("**Fuentes:** Proyecciones de población municipal por área, sexo y edad para 2024 - Censo Nacional de Población y Vivienda 2018 - Medida de Pobreza Multidimensional Municipal (IPM) 2018 - DANE. Cálculos ProColombia.") 

st.divider()

st.header('🏭 **Tejido empresarial**')
st.markdown('##### Nota: Se enfoca en personas jurídicas con sede principal en el municipio.')

if not tejido['Cod. Municipio'].isnull().all():

    c1,c2 = st.columns([20,80])
    with c1:
        # Conteo de empresas
        conteo_empresas3 = pd.pivot_table(tejido, index='Tamaño', values='Número de empresas', aggfunc='sum').reset_index()
        #conteo_empresas3 = conteo_empresas3.rename(columns={"Razón Social": "Número de empresas"})
        total_empresas = conteo_empresas3['Número de empresas'].sum()
        st.markdown('##')
        st.markdown("##### **Cantidad total de empresas**")
        st.subheader(f'{format_int(total_empresas)}')
    with c2:
        # Distribución según tamaño
        conteo_empresas3 = conteo_empresas3.sort_values(by='Número de empresas', ascending=True)
        conteo_empresas3['Participación'] = conteo_empresas3['Número de empresas'] / conteo_empresas3['Número de empresas'].sum() * 100
        text_labels = [f"{format_int(num_empresas)}<br>{format_deci(participacion)}%" 
                        for num_empresas, participacion in zip(conteo_empresas3['Número de empresas'], conteo_empresas3['Participación'])]
        fig = go.Figure([go.Bar(y=conteo_empresas3['Tamaño'], 
                                    x=conteo_empresas3['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(69, 87, 108)'),
                                    textposition='outside')])
        fig.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_empresas3)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig.update_layout(title='Distribución según tamaño', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown('##')
    st.markdown("##### **Cantidad total de empresas**")
    st.subheader(f'0')

st.markdown(
    """
    **Fuente:** Tejido Empresarial de Colombia, ProColombia.
"""
)

st.divider()

st.header('🧑‍🏭 **Panorama productivo**')

if not valor_agregado_mun['Cod. Municipio'].isnull().all():
    c1,c2 = st.columns([40,60])
    with c1:
        # Actividad económica
        st.subheader('Valor agregado 2022')
        #st.markdown("#### Valor agregado 2021")
        va = valor_agregado_mun['Valor agregado 2022'].values[0]
        peso_depto = valor_agregado_mun['Peso relativo municipal en el valor agregado departamental (%)'].values[0]
        crecimiento = valor_agregado_mun['Crecimiento 2022'].values[0]
        #st.markdown("#### Valor agregado")
        #st.subheader(f'COP {va:,.0f} miles de millones')
        st.subheader(f'COP {format_deci(va)} miles de millones')
        st.markdown(f'Participación en el valor agregado departamental: {format_deci(peso_depto)}%')
        st.markdown(f'Tasa de crecimiento 2022/2021: {format_deci(crecimiento)}%')
    with c2:
        prim = valor_agregado_mun['% Act. primarias municipio'].values[0]
        sec = valor_agregado_mun['% Act. secundarias municipio'].values[0]
        ter = valor_agregado_mun['% Act. terciarias municipio'].values[0]
        prim_formateado = format_deci(prim)
        sec_formateado = format_deci(sec)
        ter_formateado = format_deci(ter)
        fig_pib = go.Figure(data=[go.Pie(labels=['Actividades <br> primarias', 'Actividades <br> secundarias', 'Actividades <br> terciarias'], 
                                    values=[prim, sec, ter], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(255, 218, 0)', 'rgb(0, 109, 254)', 'rgb(252, 0, 81)']),
                                    customdata=[[prim_formateado, prim], [sec_formateado, sec], [ter_formateado, ter]])])
        fig_pib.update_traces(textposition='outside', 
                        textinfo='percent+label',
                        hoverinfo='label+percent',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
        fig_pib.update_layout(showlegend=False, title='Distribución por tipo de actividad', title_font=dict(size=20), annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'Año 2022',
            'showarrow': False,
            'font': {'size': 20}
        }])
        st.plotly_chart(fig_pib, use_container_width=True)
else:
    st.markdown('##### Para este municipio no hay información.') 

st.markdown("**Fuente:** Valor Agregado por municipio 2022, DANE. Cálculos ProColombia.") 

# Producción agrícola
st.subheader('Producción agropecuaria')

if not eva_mun['Código Dane municipio'].isnull().all():
    cultivos_perm = eva_mun.sort_values(by='Producción (t)', ascending=False)
    cultivos_perm['Participación'] = cultivos_perm['Producción (t)'] / cultivos_perm['Producción (t)'].sum() * 100
    cultivos_perm = cultivos_perm.head(5)
    cultivos_perm = cultivos_perm.sort_values(by='Producción (t)', ascending=True)
    text_labels = [f"{format_int(pcc_perm)}<br>{format_deci(participacion_perm)}%" 
                        for pcc_perm, participacion_perm in zip(cultivos_perm['Producción (t)'], cultivos_perm['Participación'])]
    fig_perm = go.Figure([go.Bar(y=cultivos_perm['Desagregación cultivo'], 
                                    x=cultivos_perm['Producción (t)'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(255, 218, 0)'),
                                    textposition='outside')])
    fig_perm.update_traces(hoverlabel=dict(font=dict(size=20)))
    # Calcular altura dinámica
    num_cat = len(cultivos_perm)
    base_height = 50  # Altura base por elemento
    extra_padding = 150  # Espacio adicional para márgenes y título
    dynamic_height = num_cat * base_height + extra_padding
    fig_perm.update_layout(title='Cultivos con mayor producción entre 2021 y 2023', xaxis_title='Producción 2021-2023 (Toneladas)', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
    st.plotly_chart(fig_perm, use_container_width=True)  
else:
    st.markdown('##### Para este municipio no hay información.') 

st.markdown("**Fuente:** Evaluaciones Agropecuarias Municipales 2023, Unidad de Planificación Rural Agropecuaria. Cálculos ProColombia.") 

st.subheader('Distribución empresas según cadena productiva de ProColombia')
st.markdown('##### Nota: Esta sección ayuda a identificar a qué se dedican las empresas principalmente.')

if not tejido['Cod. Municipio'].isnull().all():

    # Según cadena productiva:
    conteo_empresas1 = pd.pivot_table(tejido, index = 'Cadena productiva', values = ['Número de empresas'], aggfunc='sum').reset_index()
    conteo_empresas1 = conteo_empresas1.sort_values(by='Número de empresas', ascending=True)
    conteo_empresas1['Participación'] = conteo_empresas1['Número de empresas'] / conteo_empresas1['Número de empresas'].sum() * 100
    text_labels = [f"{format_int(num_empresas2)}<br>{format_deci(participacion2)}%" 
                        for num_empresas2, participacion2 in zip(conteo_empresas1['Número de empresas'], conteo_empresas1['Participación'])]
    fig_cad = go.Figure([go.Bar(y=conteo_empresas1['Cadena productiva'], 
                                    x=conteo_empresas1['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
    fig_cad.update_traces(hoverlabel=dict(font=dict(size=20)))
    # Calcular altura dinámica
    num_cat = len(conteo_empresas1)
    base_height = 50  # Altura base por elemento
    extra_padding = 150  # Espacio adicional para márgenes y título
    dynamic_height = num_cat * base_height + extra_padding
    fig_cad.update_layout(xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)))
    st.plotly_chart(fig_cad, use_container_width=True)

    # Cadena Agroalimentos
    agro = tejido[tejido['Cadena productiva']=="Agroalimentos"]
    if not agro['Cod. Municipio'].isnull().all():
        # Conteo de empresas
        conteo_empresas_agro = pd.pivot_table(agro, index = ['CIIU Rev 4 principal', 'Descripción CIIU principal'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        # Distribución según CIIU principal
        conteo_empresas_agro = conteo_empresas_agro.sort_values(by='Número de empresas', ascending=False)
        conteo_empresas_agro['Participación'] = conteo_empresas_agro['Número de empresas'] / conteo_empresas_agro['Número de empresas'].sum() * 100
        conteo_empresas_agro = conteo_empresas_agro[~conteo_empresas_agro['Descripción CIIU principal'].str.contains('Comercio', case=False, na=False)]
        conteo_empresas_agro = conteo_empresas_agro.head(5)
        conteo_empresas_agro = conteo_empresas_agro.sort_values(by='Número de empresas', ascending=True)
        text_labels = [f"{format_int(num_empresas_agro)}<br>{format_deci(participacion_agro)}%" 
                        for num_empresas_agro, participacion_agro in zip(conteo_empresas_agro['Número de empresas'], conteo_empresas_agro['Participación'])]
        fig_agro = go.Figure([go.Bar(y=conteo_empresas_agro['Descripción CIIU principal'], 
                                    x=conteo_empresas_agro['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
        fig_agro.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_empresas_agro)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_agro.update_layout(title='Actividades con mayor número de empresas en la cadena de Agroalimentos', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_agro, use_container_width=True)
    else:
        pass

    # Cadena Metalmecánica
    metal = tejido[tejido['Cadena productiva']=="Metalmecánica y Otras Industrias"]
    if not metal['Cod. Municipio'].isnull().all():
        # Conteo de empresas
        conteo_empresas_metal = pd.pivot_table(metal, index = ['CIIU Rev 4 principal', 'Descripción CIIU principal'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        # Distribución según CIIU principal
        conteo_empresas_metal = conteo_empresas_metal.sort_values(by='Número de empresas', ascending=False)
        conteo_empresas_metal['Participación'] = conteo_empresas_metal['Número de empresas'] / conteo_empresas_metal['Número de empresas'].sum() * 100
        conteo_empresas_metal = conteo_empresas_metal[~conteo_empresas_metal['Descripción CIIU principal'].str.contains('Comercio', case=False, na=False)]
        conteo_empresas_metal = conteo_empresas_metal.head(5)
        conteo_empresas_metal = conteo_empresas_metal.sort_values(by='Número de empresas', ascending=True)
        text_labels = [f"{format_int(num_empresas_metal)}<br>{format_deci(participacion_metal)}%" 
                        for num_empresas_metal, participacion_metal in zip(conteo_empresas_metal['Número de empresas'], conteo_empresas_metal['Participación'])]
        fig_metal = go.Figure([go.Bar(y=conteo_empresas_metal['Descripción CIIU principal'], 
                                    x=conteo_empresas_metal['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
        fig_metal.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_empresas_metal)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_metal.update_layout(title='Actividades con mayor número de empresas en la cadena de Metalmecánica y Otras Industrias', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_metal, use_container_width=True)
    else:
        pass
    
    # Cadena Químicos y Ciencias de la Vida
    quim = tejido[tejido['Cadena productiva']=="Químicos y Ciencias de la Vida"]
    if not quim['Cod. Municipio'].isnull().all():
        # Conteo de empresas
        conteo_empresas_quim = pd.pivot_table(quim, index = ['CIIU Rev 4 principal', 'Descripción CIIU principal'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        # Distribución según CIIU principal
        conteo_empresas_quim = conteo_empresas_quim.sort_values(by='Número de empresas', ascending=False)
        conteo_empresas_quim['Participación'] = conteo_empresas_quim['Número de empresas'] / conteo_empresas_quim['Número de empresas'].sum() * 100
        conteo_empresas_quim = conteo_empresas_quim[~conteo_empresas_quim['Descripción CIIU principal'].str.contains('Comercio', case=False, na=False)]
        conteo_empresas_quim = conteo_empresas_quim.head(5)
        conteo_empresas_quim = conteo_empresas_quim.sort_values(by='Número de empresas', ascending=True)
        text_labels = [f"{format_int(num_empresas_quim)}<br>{format_deci(participacion_quim)}%" 
                        for num_empresas_quim, participacion_quim in zip(conteo_empresas_quim['Número de empresas'], conteo_empresas_quim['Participación'])]
        fig_quim = go.Figure([go.Bar(y=conteo_empresas_quim['Descripción CIIU principal'], 
                                    x=conteo_empresas_quim['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
        fig_quim.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_empresas_quim)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_quim.update_layout(title='Actividades con mayor número de empresas en la cadena de Químicos y Ciencias de la Vida', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_quim, use_container_width=True)
    else:
        pass

    # Cadena Sistema Moda
    moda = tejido[tejido['Cadena productiva']=="Sistema Moda"]
    if not moda['Cod. Municipio'].isnull().all():
        # Conteo de empresas
        conteo_empresas_moda = pd.pivot_table(moda, index = ['CIIU Rev 4 principal', 'Descripción CIIU principal'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        # Distribución según CIIU principal
        conteo_empresas_moda = conteo_empresas_moda.sort_values(by='Número de empresas', ascending=False)
        conteo_empresas_moda['Participación'] = conteo_empresas_moda['Número de empresas'] / conteo_empresas_moda['Número de empresas'].sum() * 100
        conteo_empresas_moda = conteo_empresas_moda[~conteo_empresas_moda['Descripción CIIU principal'].str.contains('Comercio', case=False, na=False)]
        conteo_empresas_moda = conteo_empresas_moda.head(5)
        conteo_empresas_moda = conteo_empresas_moda.sort_values(by='Número de empresas', ascending=True)
        text_labels = [f"{format_int(num_empresas_moda)}<br>{format_deci(participacion_moda)}%" 
                        for num_empresas_moda, participacion_moda in zip(conteo_empresas_moda['Número de empresas'], conteo_empresas_moda['Participación'])]
        fig_moda = go.Figure([go.Bar(y=conteo_empresas_moda['Descripción CIIU principal'], 
                                    x=conteo_empresas_moda['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
        fig_moda.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_empresas_moda)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_moda.update_layout(title='Actividades con mayor número de empresas en la cadena de Sistema Moda', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_moda, use_container_width=True)
    else:
        pass
    
    # Cadena Industrias 4.0
    serv = tejido[tejido['Cadena productiva']=="Industrias 4.0"]
    if not serv['Cod. Municipio'].isnull().all():
        # Conteo de empresas
        conteo_empresas_serv = pd.pivot_table(serv, index = ['CIIU Rev 4 principal', 'Descripción CIIU principal'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        # Distribución según CIIU principal
        conteo_empresas_serv = conteo_empresas_serv.sort_values(by='Número de empresas', ascending=False)
        conteo_empresas_serv['Participación'] = conteo_empresas_serv['Número de empresas'] / conteo_empresas_serv['Número de empresas'].sum() * 100
        conteo_empresas_serv = conteo_empresas_serv[~conteo_empresas_serv['Descripción CIIU principal'].str.contains('Comercio', case=False, na=False)]
        conteo_empresas_serv = conteo_empresas_serv.head(5)
        conteo_empresas_serv = conteo_empresas_serv.sort_values(by='Número de empresas', ascending=True)
        text_labels = [f"{format_int(num_empresas_serv)}<br>{format_deci(participacion_serv)}%" 
                        for num_empresas_serv, participacion_serv in zip(conteo_empresas_serv['Número de empresas'], conteo_empresas_serv['Participación'])]
        fig_serv = go.Figure([go.Bar(y=conteo_empresas_serv['Descripción CIIU principal'], 
                                    x=conteo_empresas_serv['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
        fig_serv.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_empresas_serv)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_serv.update_layout(title='Actividades con mayor número de empresas en la cadena de Industrias 4.0', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_serv, use_container_width=True)
    else:
        pass
    
    # Según valor agregado empresa:

    # Bienes
    conteo_empresas2 = pd.pivot_table(tejido, index = 'Valor agregado empresa', values = ['Número de empresas'], aggfunc='sum').reset_index()
    conteo_bienes = conteo_empresas2[conteo_empresas2['Valor agregado empresa'].str.contains('Bienes', case=False, na=False)]
    if not conteo_bienes['Valor agregado empresa'].isnull().all():
        st.subheader('Distribución empresas de bienes según la intensidad tecnológica de su actividad')
        conteo_bienes = conteo_bienes.sort_values(by='Número de empresas', ascending=True)
        conteo_bienes['Participación'] = conteo_bienes['Número de empresas'] / conteo_bienes['Número de empresas'].sum() * 100
        text_labels = [f"{format_int(num_empresas_bienes)}<br>{format_deci(participacion_bienes)}%" 
                                for num_empresas_bienes, participacion_bienes in zip(conteo_bienes['Número de empresas'], conteo_bienes['Participación'])]
        fig_bienes = go.Figure([go.Bar(y=conteo_bienes['Valor agregado empresa'], 
                                            x=conteo_bienes['Número de empresas'], 
                                            text=text_labels,
                                            hoverinfo='text',
                                            orientation='h',
                                            textangle=0,
                                            marker=dict(color='rgb(252, 0, 81)'),
                                            textposition='outside')])
        fig_bienes.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_bienes)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_bienes.update_layout(xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)))
        st.plotly_chart(fig_bienes, use_container_width=True)
    else:
        pass
    
    # Servicios
    conteo_empresas2 = pd.pivot_table(tejido, index = 'Valor agregado empresa', values = ['Número de empresas'], aggfunc='sum').reset_index()
    conteo_serv = conteo_empresas2[~conteo_empresas2['Valor agregado empresa'].str.contains('Bienes', case=False, na=False)]
    if not conteo_serv['Valor agregado empresa'].isnull().all():
        st.subheader('Distribución empresas de servicios según la intensidad en conocimiento de su actividad')
        conteo_serv = conteo_serv.sort_values(by='Número de empresas', ascending=True)
        conteo_serv['Participación'] = conteo_serv['Número de empresas'] / conteo_serv['Número de empresas'].sum() * 100
        text_labels = [f"{format_int(num_empresas_serv)}<br>{format_deci(participacion_serv)}%" 
                                for num_empresas_serv, participacion_serv in zip(conteo_serv['Número de empresas'], conteo_serv['Participación'])]
        fig_serv = go.Figure([go.Bar(y=conteo_serv['Valor agregado empresa'], 
                                            x=conteo_serv['Número de empresas'], 
                                            text=text_labels,
                                            hoverinfo='text',
                                            orientation='h',
                                            textangle=0,
                                            marker=dict(color='rgb(252, 0, 81)'),
                                            textposition='outside')])
        fig_serv.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(conteo_serv)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_serv.update_layout(xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)))
        st.plotly_chart(fig_serv, use_container_width=True)
    else:
        pass
else:
    st.markdown('##')
    st.markdown("##### No se identificaron personas jurídicas con sede principal en este municipio.")

st.markdown(
    """
    **Fuente:** Tejido Empresarial de Colombia, ProColombia.
"""
)

st.divider()

st.header('👨‍💼 **Indicadores laborales**')
st.markdown('#### Censo 2018')

if not censo_mun['Cod. Municipio'].isnull().all():
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.subheader('Fuerza laboral')
        fuerza_laboral = censo_mun['FT'].values[0]
        fuerza_formateado = format_int(fuerza_laboral)
        st.markdown(f'#### {fuerza_formateado} personas')
    with c2:
        st.subheader('Ocupados')
        ocupados = censo_mun['OCU'].values[0]
        ocu_formateado = format_int(ocupados)
        st.markdown(f'#### {ocu_formateado} personas')
    with c3:
        st.subheader('Tasa de ocupación')
        tasa_ocupacion = censo_mun['Tasa ocupación'].values[0]
        tasa_ocu_formateado = format_deci(tasa_ocupacion)
        st.markdown(f'#### {tasa_ocu_formateado}%')
    with c4:
        st.subheader('Tasa de desempleo')
        tasa_desempleo = censo_mun['Tasa desempleo'].values[0]
        tasa_des_formateado = format_deci(tasa_desempleo)
        st.markdown(f'#### {tasa_des_formateado}%')
else:
    st.markdown('##### Para este municipio no hay información.') 

st.markdown("**Fuente:** Censo Nacional de Población y Vivienda 2018, DANE. Cálculos ProColombia.") 

st.divider()

st.header('🧑‍🎓 **Talento humano**')

st.subheader('Nivel educativo de la población de 20 años o más')

# Nivel educativo
if not censo_mun['Cod. Municipio'].isnull().all():
    #st.markdown("######")
    #st.subheader('Nivel educativo de la población')
    #st.markdown("####")
    media = censo_mun['% pobl. con educación media municipio'].values[0]
    tecnica = censo_mun['% pobl. con edu. técnica/tecnología municipio'].values[0]
    pre = censo_mun['% pobl. con pregrado municipio'].values[0]
    pos = censo_mun['% pobl. con posgrado municipio'].values[0]
    resto = 100 - media - tecnica - pre - pos
    media_formateado = format_deci(media)
    tecnica_formateado = format_deci(tecnica)
    pre_formateado = format_deci(pre)
    pos_formateado = format_deci(pos)
    resto_formateado = format_deci(resto)
    fig_edu = go.Figure(data=[go.Pie(labels=['Educación <br> media', 'Educación <br> técnica/tecnología', 'Pregrado', 'Posgrado', 'Resto <br> de <br> población <br> de <br> 20 años <br> o más'], 
                                    values=[media, tecnica, pre, pos, resto], 
                                    hole=0.5,  
                                    marker=dict(colors=['rgb(255, 218, 0)', 'rgb(0, 109, 254)', 'rgb(252, 0, 81)', 'rgb(106, 124, 133)', 'rgb(69, 87, 108)']),
                                    customdata=[[media_formateado, media], [tecnica_formateado, tecnica], [pre_formateado, pre], [pos_formateado, pos], [resto_formateado, resto]])])
    fig_edu.update_traces(textposition='outside', 
                        textinfo='percent+label',
                        hoverinfo='label+percent',
                        textfont=dict(size=16),
                        hoverlabel=dict(font=dict(size=20)),
                        hovertemplate='%{label}: %{customdata[0][0]}%',
                        texttemplate='%{label}: %{customdata[0]}%')
    fig_edu.update_layout(showlegend=False, annotations=[
        {
            'x': 0.5,
            'y': 0.5,
            'xanchor': 'center',
            'yanchor': 'middle',
            'text': 'Censo 2018',
            'showarrow': False,
            'font': {'size': 20}
        }])
    st.plotly_chart(fig_edu, use_container_width=True)
else:
    st.markdown("####")
    st.markdown('##### Para este municipio no hay información.') 

st.markdown("**Fuente:** Censo Nacional de Población y Vivienda 2018, DANE. Cálculos ProColombia.") 

st.subheader('Graduados en programas de educación superior ofrecidos en el municipio entre 2018 y 2022')
#st.markdown("#### **Graduados en programas de educación superior ofrecidos en el territorio entre 2018 y 2022**")

if not grad_mun['CÓDIGO DEL MUNICIPIO (PROGRAMA)'].isnull().all():
    c1,c2 = st.columns([20,80])
    with c1:
        total_grad = grad_mun['GRADUADOS'].sum()
        st.markdown('##')
        st.markdown("##### **Total graduados**")
        st.subheader(f'{format_int(total_grad)}')
    with c2:
        grad_mun = grad_mun.sort_values(by='GRADUADOS', ascending=True)
        grad_mun['Participación'] = grad_mun['GRADUADOS'] / grad_mun['GRADUADOS'].sum() * 100
        grad_mun['CINE'] = grad_mun['CINE CAMPO AMPLIO'].str.capitalize()
        text_labels = [f"{format_int(num_grad)}<br>{format_deci(participacion)}%" 
                        for num_grad, participacion in zip(grad_mun['GRADUADOS'], grad_mun['Participación'])]
        fig_grad = go.Figure([go.Bar(y=grad_mun['CINE'], 
                                    x=grad_mun['GRADUADOS'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
        fig_grad.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cat = len(grad_mun)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cat * base_height + extra_padding
        fig_grad.update_layout(title='Distribución según rama de conocimiento', xaxis_title='Graduados 2018-2022', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_grad, use_container_width=True)
else:
    st.markdown("####")
    st.markdown('##### Para este municipio no hay información.') 

st.markdown("**Fuente:** Sistema Nacional de Información de la Educación Superior - Ministerio de Educación Nacional. Cálculos ProColombia.") 

st.divider()

# Tejido exportador
st.header('🚢 **Exportaciones**')

st.subheader('Empresas con sede principal en el municipio que realizaron al menos una exportación en los últimos 10 años (2013-2022)')

exportadoras = tejido[tejido['Tipo* ult 10 años']!="No exportó ult. 10 años"]
if not exportadoras['Cod. Municipio'].isnull().all():
    c1,c2 = st.columns([20,80])
    with c1:
        # Conteo de empresas
        conteo_empresas4 = pd.pivot_table(exportadoras, index = ['Cadena* ult 10 años'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        total_exportadoras = conteo_empresas4['Número de empresas'].sum()
        st.markdown('##')
        st.markdown("##### **Cantidad total de empresas**")
        st.subheader(f'{format_int(total_exportadoras)}')
    with c2:
        # Distribución según cadena que más exporta
        conteo_empresas4 = conteo_empresas4.sort_values(by='Número de empresas', ascending=True)
        conteo_empresas4['Participación'] = conteo_empresas4['Número de empresas'] / conteo_empresas4['Número de empresas'].sum() * 100
        text_labels = [f"{format_int(num_empresas4)}<br>{format_deci(participacion4)}%" 
                        for num_empresas4, participacion4 in zip(conteo_empresas4['Número de empresas'], conteo_empresas4['Participación'])]
        fig_expo = go.Figure([go.Bar(y=conteo_empresas4['Cadena* ult 10 años'], 
                                    x=conteo_empresas4['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(252, 0, 81)'),
                                    textposition='outside')])
        fig_expo.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cadenas = len(conteo_empresas4)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cadenas * base_height + extra_padding
        fig_expo.update_layout(title='Distribución según la cadena productiva por la que más exportó la empresa', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_expo, use_container_width=True)
    
    # Distribución según subsector que más exporta
    conteo_empresas_subsector = pd.pivot_table(exportadoras, index = ['Subsector* ult 10 años'], values = ['Número de empresas'], aggfunc='sum').reset_index()
    conteo_empresas_subsector['Participación'] = conteo_empresas_subsector['Número de empresas'] / conteo_empresas_subsector['Número de empresas'].sum() * 100
    conteo_empresas_subsector = conteo_empresas_subsector.sort_values(by='Número de empresas', ascending=False)
    conteo_empresas_subsector = conteo_empresas_subsector.head(10)
    conteo_empresas_subsector = conteo_empresas_subsector.sort_values(by='Número de empresas', ascending=True)
    text_labels = [f"{format_int(num_empresas_subsector)}<br>{format_deci(participacion_subsector)}%" 
                        for num_empresas_subsector, participacion_subsector in zip(conteo_empresas_subsector['Número de empresas'], conteo_empresas_subsector['Participación'])]
    fig_expo_subsector = go.Figure([go.Bar(y=conteo_empresas_subsector['Subsector* ult 10 años'], 
                                    x=conteo_empresas_subsector['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(252, 0, 81)'),
                                    textposition='outside')])
    fig_expo_subsector.update_traces(hoverlabel=dict(font=dict(size=20)))
    # Calcular altura dinámica
    num_subsector = len(conteo_empresas_subsector)
    base_height = 50  # Altura base por elemento
    extra_padding = 150  # Espacio adicional para márgenes y título
    dynamic_height = num_subsector * base_height + extra_padding
    fig_expo_subsector.update_layout(title='Principales subsectores por el que más exportó la empresa', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
    st.plotly_chart(fig_expo_subsector, use_container_width=True)
    
    st.markdown('#### Principales productos o servicios que más exportó la empresa')
    conteo_empresas_posara = pd.pivot_table(exportadoras, index = ['Posara* ult 10 años', 'Descripcion posara* ult 10 años'], values = ['Número de empresas'], aggfunc='sum').reset_index()
    conteo_empresas_posara['Participación'] = conteo_empresas_posara['Número de empresas'] / conteo_empresas_posara['Número de empresas'].sum() * 100
    conteo_empresas_posara = conteo_empresas_posara.sort_values(by='Número de empresas', ascending=False)
    conteo_empresas_posara = conteo_empresas_posara.head(10)
    #conteo_empresas_posara = conteo_empresas_posara.sort_values(by='Número de empresas', ascending=True)
    # Formatear las columnas para una mejor visualización
    conteo_empresas_posara['Número de empresas'] = conteo_empresas_posara['Número de empresas'].apply(format_int)
    conteo_empresas_posara['Participación'] = conteo_empresas_posara['Participación'].apply(format_deci)
    # Renombrar las columnas si es necesario
    conteo_empresas_posara = conteo_empresas_posara.rename(columns={
    'Posara* ult 10 años': 'Posición arancelaria',
    'Descripcion posara* ult 10 años': 'Descripcion',
    'Participación': 'Participación (%)'
    })
    # Mostrar la tabla en Streamlit
    st.dataframe(conteo_empresas_posara, use_container_width=True, hide_index=True)

else:
    st.markdown('##')
    st.markdown("##### **Cantidad total de empresas**")
    st.subheader(f'0')

st.markdown(
    """
    **Fuente:** Tejido Empresarial de Colombia, ProColombia.
"""
)

st.divider()

# Instalados
st.header('💵 **Inversión**')

st.subheader('Empresas con sede principal en el municipio identificadas como sucursal de sociedad extranjera')

ied = tejido[tejido['Sucursal sociedad extranjera']=="Si"]
if not ied['Cod. Municipio'].isnull().all():
    c1,c2 = st.columns([20,80])
    with c1:
        # Conteo de empresas
        conteo_empresas5 = pd.pivot_table(ied, index = ['Cadena productiva'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        total_ied = conteo_empresas5['Número de empresas'].sum()
        st.markdown('##')
        st.markdown("##### **Cantidad total de empresas**")
        st.subheader(f'{format_int(total_ied)}')
    with c2:
        # Distribución según cadena
        conteo_empresas5 = conteo_empresas5.sort_values(by='Número de empresas', ascending=True)
        conteo_empresas5['Participación'] = conteo_empresas5['Número de empresas'] / conteo_empresas5['Número de empresas'].sum() * 100
        text_labels = [f"{format_int(num_empresas5)}<br>{format_deci(participacion5)}%" 
                        for num_empresas5, participacion5 in zip(conteo_empresas5['Número de empresas'], conteo_empresas5['Participación'])]
        fig_ied = go.Figure([go.Bar(y=conteo_empresas5['Cadena productiva'], 
                                    x=conteo_empresas5['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(0, 109, 254)'),
                                    textposition='outside')])
        fig_ied.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cadenas = len(conteo_empresas5)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cadenas * base_height + extra_padding
        fig_ied.update_layout(title='Distribución según cadena productiva', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_ied, use_container_width=True)
else:
    st.markdown('##')
    st.markdown("##### **Cantidad total de empresas**")
    st.subheader(f'0')

st.markdown(
    """
    **Fuente:** Tejido Empresarial de Colombia, ProColombia.
"""
)

st.divider()

# Turismo
st.header('🛬 **Turismo**')

st.subheader('Empresas con sede principal en el municipio relacionadas con actividades de turismo')

turismo = tejido[tejido['Cadena productiva']=="Turismo"]
if not turismo['Cod. Municipio'].isnull().all():
    c1,c2 = st.columns([20,80])
    with c1:
        # Conteo de empresas
        conteo_empresas6 = pd.pivot_table(turismo, index = ['CIIU Rev 4 principal', 'Descripción CIIU principal'], values = ['Número de empresas'], aggfunc='sum').reset_index()
        total_tur = conteo_empresas6['Número de empresas'].sum()
        st.markdown('##')
        st.markdown("##### **Cantidad total de empresas**")
        st.subheader(f'{format_int(total_tur)}')
    with c2:
        # Distribución según CIIU principal
        conteo_empresas6 = conteo_empresas6.sort_values(by='Número de empresas', ascending=True)
        conteo_empresas6['Participación'] = conteo_empresas6['Número de empresas'] / conteo_empresas6['Número de empresas'].sum() * 100
        text_labels = [f"{format_int(num_empresas6)}<br>{format_deci(participacion6)}%" 
                        for num_empresas6, participacion6 in zip(conteo_empresas6['Número de empresas'], conteo_empresas6['Participación'])]
        fig_tur = go.Figure([go.Bar(y=conteo_empresas6['Descripción CIIU principal'], 
                                    x=conteo_empresas6['Número de empresas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(255, 218, 0)'),
                                    textposition='outside')])
        fig_tur.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_cadenas = len(conteo_empresas6)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_cadenas * base_height + extra_padding
        fig_tur.update_layout(title='Distribución según CIIU principal', xaxis_title='Número de empresas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_tur, use_container_width=True)
else:
    st.markdown('##')
    st.markdown("##### **Cantidad total de empresas**")
    st.subheader(f'0')

st.markdown(
    """
    **Fuente:** Tejido Empresarial de Colombia, ProColombia.
"""
)

st.subheader('Llegada de visitantes no residentes 2023')

if not tec_mun['CODIGO'].isnull().all():
    c1,c2 = st.columns([30,70])
    with c1:
        tec_mun['Mes'] = pd.to_datetime(tec_mun['Mes'])
        llegadas2023 = tec_mun[tec_mun['Mes'].dt.year == 2023]['LLegadas'].sum()
        crecimiento = (tec_mun[tec_mun['Mes'].dt.year == 2023]['LLegadas'].sum() / tec_mun[tec_mun['Mes'].dt.year == 2022]['LLegadas'].sum() - 1)*100
        st.subheader(f'{format_int(llegadas2023)} personas')
        st.markdown(f'Tasa de crecimiento 2023/2022: {format_deci(crecimiento)}%')
    with c2:
        # Distribución por país de residencia
        tec_mun_2023 = tec_mun[tec_mun['Mes'].dt.year == 2023]
        tec_mun2 = pd.pivot_table(tec_mun_2023, index = ['Nombre Oee país residencia'], values = ['LLegadas'], aggfunc='sum').reset_index()
        tec_mun2['Participación'] = tec_mun2['LLegadas'] / tec_mun2['LLegadas'].sum() * 100
        tec_mun2 = tec_mun2.sort_values(by='LLegadas', ascending=False)
        tec_mun2 = tec_mun2.head(10)
        tec_mun2 = tec_mun2.sort_values(by='LLegadas', ascending=True)
        text_labels = [f"{format_int(lleg_pais)}<br>{format_deci(lleg_part)}%" 
                        for lleg_pais, lleg_part in zip(tec_mun2['LLegadas'], tec_mun2['Participación'])]
        fig_tur_pais = go.Figure([go.Bar(y=tec_mun2['Nombre Oee país residencia'], 
                                    x=tec_mun2['LLegadas'], 
                                    text=text_labels,
                                    hoverinfo='text',
                                    orientation='h',
                                    textangle=0,
                                    marker=dict(color='rgb(255, 218, 0)'),
                                    textposition='outside')])
        fig_tur_pais.update_traces(hoverlabel=dict(font=dict(size=20)))
        # Calcular altura dinámica
        num_paises = len(tec_mun2)
        base_height = 50  # Altura base por elemento
        extra_padding = 150  # Espacio adicional para márgenes y título
        dynamic_height = num_paises * base_height + extra_padding
        fig_tur_pais.update_layout(title='Distribución según país de residencia', xaxis_title='Número de llegadas', height=dynamic_height, font=dict(size=16), xaxis=dict(tickfont=dict(size=16)), yaxis=dict(tickfont=dict(size=16)), title_font=dict(size=20))
        st.plotly_chart(fig_tur_pais, use_container_width=True)
else:
    st.markdown('##')
    st.markdown("##### **Cantidad total de llegadas**")
    st.subheader(f'0')

st.markdown(
    """
    **Fuente:** Migración Colombia.
"""
)

st.divider()

st.subheader('Menú')

st.page_link("pages/2-Metodologia.py", label="👆 :blue[**Metodología**]")
st.markdown("""Conoce aspectos metodológicos relevantes sobre cómo se construyó la infomación de los municipios.""")
st.page_link("app.py", label="👆 :blue[**Página principal**]")
st.markdown("""Regresa a la página principal.""")

st.divider()