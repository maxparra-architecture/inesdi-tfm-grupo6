import streamlit as st
import os
#from PIL import Image
from menu import menu

# Rutas
path_cwd = os.getcwd()
path_data = path_cwd + "/data/"
path_img  = path_cwd + "/img/"
youtube_url = "https://www.youtube.com/embed/r5oN0VRs6BY?autoplay=1&mute=1&modestbranding=1&showinfo=0&controls=0&loop=1&playlist=r5oN0VRs6BY"

# Configuración de la página
st.set_page_config(page_title="Perfiles Municipios de Colombia", page_icon = '🌎', layout="wide",  
                   initial_sidebar_state="expanded")
menu()

st.markdown("<h1 style='text-align: center;'>Perfiles Municipios de Colombia</h1>", unsafe_allow_html=True)

st.divider()

# Insertar el video en el dashboard usando HTML y CSS
video_html = f"""
<div style="position: relative; padding-bottom: 30%; height: 0; overflow: hidden; max-width: 100%; height: auto; margin-bottom: 25px;">
    <iframe src="{youtube_url}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>
"""

# Mostrar el video usando componentes HTML
st.markdown(video_html, unsafe_allow_html=True)

st.markdown(
    """
    Esta herramienta permite seleccionar cualquier municipio de Colombia y descubrir información relevante del territorio 
    como ubicación geográfica, si es territorio PDET o ZOMAC, si tiene menos de 200.000 habitantes, población diferencial 
    residente, vocación productiva, talento humano, tejido empresarial total y tejido empresarial relacionado con 
    exportaciones, inversión extranjera directa y turismo internacional.
    """)

st.markdown(
    """
    ##### **Beneficios:**
    """)

st.markdown(
    """
    Conocer a mayor profundidad nuestros territorios, sus características y lo que tienen para ofrecer es 
    fundamental para la formulación de políticas públicas y la promoción de exportaciones no 
    minero-energéticas, inversión extranjera directa y turismo internacional.
    """)

st.markdown(
    """
    Esta herramienta además representa un gran beneficio para los colaboradores de ProColombia, ya que 
    consultar e integrar todas las fuentes necesarias para caracterizar el territorio puede tomarles un mes 
    o más, mientras que acá pueden consultar información diversa sobre los municipios en unos pocos segundos. 
    Esto es posible gracias a que hemos implementado el uso de tecnologías como Snowflake, Python y Streamlit 
    y técnicas de minería de datos y Procesamiento de Lenguaje Natural.
    """)

st.markdown(
    """
    ##### **Alcance:**
    """)

st.markdown(
    """
    Es importante tener en cuenta algunas limitaciones de los datos:
    """)

st.markdown(
    """
    - La información estadística a nivel municipal es rezagada y en varios casos corresponde a la información 
    levantada en el Censo Nacional de Población y Vivienda de 2018. 
    """)

st.markdown(
    """
    - El tejido empresarial que se muestra está enfocado en personas jurídicas que tienen su sede principal en 
    el territorio. 
    """)

st.markdown(
    """
    - Respecto a las exportaciones, las estadísticas de Colombia solo identifican el departamento de origen, 
    no el municipio específico. Por este motivo, en esta herramienta se analizan las exportaciones que 
    realizan las empresas con sede principal en el territorio, aunque la producción y exportación podrían 
    haber ocurrido en otro municipio. 
    """)

st.divider()

st.subheader('Menú')

st.page_link("pages/1-Perfil.py", label="👆 :blue[**Perfiles municipios**]")
st.markdown("""Selecciona un municipio de Colombia y descubre toda la información disponible.""")
st.page_link("pages/2-Metodologia.py", label="👆 :blue[**Metodología**]")
st.markdown("""Conoce aspectos metodológicos relevantes sobre cómo se construyó la infomación de los municipios.""")

st.divider()

