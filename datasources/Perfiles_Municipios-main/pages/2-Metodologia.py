import streamlit as st
import os
from menu import menu

# Rutas
path_cwd = os.getcwd()
path_data = path_cwd + "/data/"
path_img  = path_cwd + "/img/"

# Configuración página web
st.set_page_config(page_title="Perfiles Municipios de Colombia", page_icon = '🌎', layout="wide",  initial_sidebar_state="expanded")
menu()

#st.title('Metodología')
#st.markdown("<h1 style='text-align: center;'>Metodología</h1>", unsafe_allow_html=True)
st.image(path_img + "metodologia.png", use_column_width=True)

st.divider()

st.markdown(
    """
    A continuación los invitamos a leer la metodología con el fin de entender e interpretar mejor los datos, conocer sus fuentes 
    y sus limitantes.
    """) 

with st.expander("🌐 **Ubicación geográfica**"):
    st.markdown(
        """
        Se tomó la latitud y longitud de cada municipio de Colombia, disponible en la Codificación de la División Político 
        Administrativa de Colombia - DIVIPOLA Marzo 2024 del DANE.
        """)

with st.expander('🔍 **Información general**'):

    st.markdown(
        """
        En esta sección, se identificó si el municipio hace parte de 
        los Programas de Desarrollo con Enfoque Territorial (PDET), si pertenece a las Zonas Más Afectadas 
        por el Conflicto Armado (ZOMAC) y el tamaño de su población, con el fin de identificar si es un territorio de menos
        de 200.000 habitantes.
        - **Programas de Desarrollo con Enfoque Territorial (PDET):** son instrumentos de gestión y planificación para 
        priorizar la implementación de los planes sectoriales y programas dentro de la Reforma Rural Integral. Los 
        municipios priorizados para estos programas se encuentran plasmados en la Ley 893 de 2017 y 
        corresponden a 170 municipios considerados como los más afectados por la violencia, la pobreza, las economías 
        ilícitas y la debilidad institucional. Mediante la implementación de la Reforma Rural Integral se buscan 
        sentar las bases para la transformación del campo, generar desarrollo social, reducir los índices de pobreza 
        y crear desarrollo integral de la sociedad mediante proyectos de inversión que generen bienestar para la 
        población. Además existen incentivos tributarios para fomentar el crecimiento económico en dichos territorios. 
        - **Zonas Más Afectadas por el Conflicto Armado (ZOMAC):** como su nombre lo dice, son un conjunto de municipios 
        que agrupa las zonas del país más afectadas por el conflicto armado. Las empresas ubicadas en estas zonas 
        cuentan con beneficios e incentivos tributarios, los cuales están descritos en los artículos 235 al 237 de la 
        Ley 2819 de 2016 y el Decreto 1650 de 2017.
    """
    )

    st.markdown("**Fuente:** OXI, Consultoría Obras por Impuestos (https://www.obrasximpuestos.com/).") 

with st.expander('👨‍👩‍👧‍👦 **Población diferencial**'):

    st.markdown(
        """
        Con el fin de caracterizar la población diferencial residente en cada municipio de Colombia, se recurrió a las siguientes 
        fuentes de información:
        - Proyecciones de población municipal por área, sexo y edad para 2024, DANE: permitió identificar el porcentaje de la población 
        que corresponde a mujeres y a jóvenes entre 14 y 28 años (Ley 1622 de 2013).
        - Censo Nacional de Población y Vivienda 2018, DANE: permitió identificar el porcentaje de población con discapacidad y 
        el porcentaje de población correspondiente a grupos étnicos 
        (Indígena, Gitano o Rom, Raizal del Archipiélago de San Andrés, Providencia y Santa Catalina, Palenquero de San 
        Basilio, Negro, Mulato, Afrodescendiente, Afrocolombiano).
        - Medida de Pobreza Multidimensional Municipal 2018, DANE: permitió identificar el porcentaje 
        de la población en situación de pobreza y el porcentaje de hogares que tienen ocupados informales.
    """
    )


with st.expander('🏭 **Tejido empresarial**'):

    st.markdown(
        """
        Se consultó la base "Tejido Empresarial de Colombia" construída por la Coordinación de Analítica - Gerencia de Inteligencia Comercial
        de ProColombia. Esta base tuvo en cuenta las siguientes fuentes: 
        - Registro Único Empresarial y Social (RUES) con corte a mayo de 2023, que incluyó empresas con renovación de matrícula mercantil desde 
        el año 2019 en adelante, clasificadas como “sociedad o persona jurídica principal” y en estado “activa”.
        - Directorio empresarial del DANE con corte a abril 2023.
        - Las 10.000 empresas más grandes de Colombia, Superintendencia de Sociedades (2021).
        - Base de exportaciones de bienes, DANE-DIAN (2013-2022).
        - CRM de ProColombia (2013-2022).
    """
    )

    st.markdown(
        """
        Es importante aclarar que esta base se enfoca en personas jurídicas principales (no tiene en cuenta sucursales). Aunque también se 
        encuentran algunas personas naturales provenientes de la base de exportaciones de bienes y/o del CRM de ProColombia, estos casos solo 
        representan el 2% del total de empresas que se identificaron para Colombia.
    """
    )

    st.markdown(
        """
        En el proceso de construcción de la base "Tejido Empresarial de Colombia", cada una de las cinco fuentes previamente mencionadas 
        recibió un tratamiento que consistió en la limpieza interna, con el propósito de 
        asegurar la precisión y la calidad de la información contenida. Para ello se realizó decodificación de variables, 
        unificación de unidades de medida, unificación de categorías y se analizaron y ajustaron valores extremos, datos inconsistentes y
        duplicados. Además, se unificó el NIT y la razón social de las empresas utilizando algoritmos avanzados de 
        procesamiento de lenguaje natural (PLN).
    """
    )

    st.markdown(
        """
        Las cinco fuentes se unieron con el fin de consolidar un registro amplio del tejido empresarial. Se efectuó la 
        identificación de empresas presentes en múltiples fuentes a través de su NIT o mediante el algoritmo de PLN y se 
        tomaron las siguientes decisiones frente a variables existentes en varias fuentes de datos:
        - En cuanto a la asignación de la razón social, se inició con el nombre 
        proporcionado por el RUES; en caso de no disponer de este, se recurrió al suministrado por la Superintendencia de 
        Sociedades; de no estar, se tomó el del Directorio empresarial del DANE, seguido por la base de exportaciones de 
        bienes DANE-DIAN y finalizando con la razón social registrada en el CRM de ProColombia.
        - Respecto a los datos de ubicación geográfica y actividad económica de las empresas, la primacía la tuvo el RUES. Solo si 
        la empresa no figuraba en el RUES, se recurrió a los datos proporcionados por la Superintendencia de Sociedades y, 
        en ausencia de información en ambos registros, se optó por la información contenida en el Directorio empresarial 
        del DANE.
        - Para la información de contacto, que incluye dirección, teléfono y correo electrónico, se realizó una integración 
        y consolidación de la información proveniente del RUES, la Superintendencia de Sociedades y el Directorio 
        empresarial del DANE, para garantizar una base de datos de contacto completa y fiable.
        - En lo que atañe a los datos de activos, ingresos operacionales, utilidad y la identificación de si es sucursal de 
        sociedad extranjera, se priorizó la información validada por la Superintendencia de Sociedades. En los casos donde 
        no estaba disponible esta fuente, se tomaron los datos registrados en el RUES. Cabe destacar que el Directorio 
        empresarial del DANE no dispone de estas variables.
        - Para las cifras de exportaciones de bienes se utilizó la base de exportaciones de bienes DANE-DIAN, mientras que 
        para las exportaciones de servicios se recurrió a la información de negocios provista por el CRM de ProColombia.
    """
    )

    st.markdown("**Tamaño de la empresa:**")

    st.markdown(
        """
        En la base "Tejido Empresarial de Colombia" se determinó el tamaño de las empresas con base en el Decreto 957 de 2019, 
        donde la clasificación se realiza según la actividad económica y el nivel de ingresos de la empresa:
        - Con base en el decreto, se hizo el cálculo para las 10.000 empresas más grandes de Colombia, fuente Superintendencia de Sociedades.
        - Para el resto de las empresas, se tomó el tamaño informado por el RUES con corte a mayo de 2023, donde cada cámara de comercio se 
        encarga de realizar el cálculo con base en el Decreto 957 de 2019.
    """
    )

    st.markdown(
        """
        La categoría "No determinado" corresponde a empresas que no tienen su clasificación de tamaño en el RUES o que provienen de 
        fuentes de información como Directorio Empresarial del DANE, Base de exportaciones de bienes DANE-DIAN o CRM de ProColombia 
        donde no se encuentra información para realizar la clasificación por tamaño.
    """
    )

with st.expander('🧑‍🏭 **Panorama productivo**'):

    st.markdown(
        """
        Con el fin de identificar la vocación productiva de cada municipio, se consultaron las siguientes fuentes de información:
    """
    )

    st.subheader('Valor agregado 2022')

    st.markdown(
        """
        Se consultó el "Valor Agregado por municipio 2022" del DANE, con el objetivo de conocer el nivel de valor agregado en pesos colombianos, 
        su participación en el total del departamento, su tasa de crecimiento respecto al año anterior y cómo la producción del territorio se 
        distribuyó en actividades primarias, secundarias y terciarias.
        - Actividades primarias: incluye las actividades de agricultura, ganadería, silvicultura y 
        pesca; y explotación de minas y canteras.
        - Actividades secundarias: incluye las actividades de industrias manufactureras y construcción.
        - Actividades terciarias: incluye las actividades de electricidad, gas y agua; comercio; 
        reparación de vehículos automotores; transporte; alojamiento y servicios de comida; información 
        y comunicaciones; actividades financieras y de seguros; actividades inmobiliarias; actividades 
        profesionales, científicas y técnicas; actividades de servicios administrativos y de apoyo; 
        administración pública ; educación; salud; actividades artísticas, de entretenimiento y 
        recreación; actividades de los hogares individuales.
    """
    )

    st.subheader('Producción agropecuaria')

    st.markdown(
        """
        Se consultaron las "Evaluaciones Agropecuarias Municipales 2023" de la Unidad de Planificación Rural Agropecuaria (UPRA) para conocer 
        cuáles fueron los cultivos con mayor producción en toneladas entre 2021 y 2023 y así conocer la oferta agrícola del municipio.
    """
    )

    st.subheader('Distribución empresas según cadena productiva de ProColombia')

    st.markdown(
        """
        Se consultó la base "Tejido Empresarial de Colombia" de ProColombia, en la cual a cada empresa se le asigna una cadena productiva según 
        la actividad económica principal a la que se dedica. Para ello, se construyó una correlativa con los códigos CIIU Revisión 4 relevantes 
        para cada cadena productiva de ProColombia, con el apoyo de los asesores de la Gerencia de Inteligencia Comercial y de la Vicepresidencia 
        de Exportaciones. Esta información permite identificar a qué cadenas productivas se dedican las empresas con sede principal en el 
        municipio.
    """)

    st.markdown(
        """
        Adicionalmente, con base en el "Tejido Empresarial de Colombia", se identificaron las principales 5 actividades económicas 
        con mayor número de empresas en cada cadena productiva. Esto permite ver a qué actividades se dedican mayoritariamente las empresas con 
        sede principal en el municipio.
    """)

    st.subheader('Distribución empresas de bienes según la intensidad tecnológica de su actividad')

    st.markdown(
        """
        Se consultó el "Tejido Empresarial de Colombia" de ProColombia donde se tuvo en cuenta la actividad económica principal de cada 
        empresa para clasificarlas en: 
        - **Bienes primarios:** Se refiere al cultivo de productos agrícolas, cría de animales, pesca y extracción de minerales y otros recursos directamente de la naturaleza.
        - **Bienes de tecnología baja:** Incluye la producción de alimentos procesados, madera, muebles, papel, textiles, ropa, calzado y joyas.
        - **Bienes de tecnología media-baja:** Comprende productos de la refinación del petróleo, mezcla de combustibles, llantas, neumáticos, artículos de plástico y caucho, productos de vidrio, cerámica y porcelana, materiales para la construcción y productos elaborados de metal.
        - **Bienes de tecnología media-alta:** Incluye fabricación de sustancias y productos químicos, maquinaria y equipo, aparatos de uso doméstico, vehículos, motocicletas, barcos, locomotoras y aeronaves.
        - **Bienes de tecnología alta:** Incluye fabricación de productos farmacéuticos, componentes y tableros electrónicos, computadoras, equipos de comunicación, aparatos electrónicos, equipo de medición e instrumentos ópticos.
    """
    )

    st.markdown(
        """
        Esta clasificación se construyó con base en el documento "Eurostat indicators on High-tech industry and Knowledge – intensive services". 
    """
    )

    st.subheader('Distribución empresas de servicios según la intensidad en conocimiento de su actividad')

    st.markdown(
        """
        Se consultó el "Tejido Empresarial de Colombia" de ProColombia donde se tuvo en cuenta la actividad económica principal de cada 
        empresa para clasificarlas en:
        - **Servicios menos intensivos en conocimiento:** Comprende actividades como comercio, mantenimiento, reparación, transporte terrestre, almacenamiento, actividades postales, alojamiento, alimentación, agencias de viaje y actividades inmobiliarias. 
        - **Servicios de mercado intensivos en conocimiento:** Comprende servicios como transporte aéreo y acuático, actividades legales, contables, arquitectura, ingeniería, publicidad, investigación de mercados, BPO.
        - **Servicios financieros intensivos en conocimiento:** Se enfoca en actividades financieras y de seguros.
        - **Otros servicios intensivos en conocimiento:** Engloba una gama amplia de actividades como editorial, salud, educación, administración pública, defensa, arte, entretenimiento y recreación.
        - **Servicios de alta tecnología intensivos en conocimiento:** Incluye actividades relacionadas con la producción de medios audiovisuales, grabación de sonido y música, telecomunicaciones, software y servicios TI, servicios de información e investigación y desarrollo científico.
        - **Servicios no clasificados:** Relacionados con servicios públicos y construcción.
    """
    )

    st.markdown(
        """
        Esta clasificación se construyó con base en el documento "Eurostat indicators on High-tech industry and Knowledge – intensive services". 
    """
    )

with st.expander('🧑‍🎓 **Talento humano**'):

    st.subheader('Nivel educativo de la población de 20 años o más')

    st.markdown(
        """Se consultó el "Censo Nacional de Población y Vivienda 2018" del DANE, el cual permitió identificar el
        nivel educativo más alto alcanzado por la población de 20 años o más.
        """)

    st.subheader('Graduados en programas de educación superior ofrecidos en el municipio entre 2018 y 2022')

    st.markdown(
        """Se consultó el "Tablero de Graduados en Colombia 2001 – 2022", elaborado por la Coordinación de Analítica - Gerencia de Inteligencia 
        Comercial de ProColombia, el cual utiliza como fuente el Sistema Nacional de Información de la Educación Superior del Ministerio de 
        Educación Nacional.
        """)

    st.markdown(
        """Específicamente, se calculó la suma de graduados 2018-2022 de programas técnicos, tecnológicos, pregrado y posgrado ofrecidos en cada 
        municipio. De esta forma es posible identificar, para cada territorio, el número de personas que se graduaron de educación superior y 
        en qué ramas del conocimiento se formaron principalmente.
        """)

    st.markdown(
        """
        Es importante tener en cuenta que los datos se refieren a las personas que se graduaron en programas ofrecidos en el municipio y que no 
        necesariamente estos graduados residen en el mismo.
        """)

    st.markdown(
        """
        En el "Tablero de Graduados en Colombia 2001 – 2022", los programas académicos se clasificaron teniendo en cuenta el campo amplio 
        al que pertenecen de la Clasificación Internacional 
        Normalizada de la Educación - Campos de Educación y Formación (CINE) - 2013 Adaptada a Colombia. La UNESCO define un campo como la 
        esfera amplia, la rama o el área de contenido cubierto por un programa de educación. Teniendo en cuenta lo anterior, las categorías 
        consideradas fueron las siguientes:
        - **Administración de Empresas y Derecho:** se enfoca en la administración de empresas, comercio, marketing, contabilidad, finanzas, 
        y recursos humanos, además de abarcar todas las ramas del derecho.
        - **Agropecuario, Silvicultura, Pesca y Veterinaria:** trata sobre la producción agrícola, agronomía, ciencias del suelo, horticultura, 
        gestión forestal, pesca comercial y acuicultura, así como la medicina veterinaria y el cuidado de animales.
        - **Arte y Humanidades:** incluye las artes visuales y escénicas, música, diseño, cine, fotografía, y artesanías. En humanidades, abarca 
        historia, filosofía, arqueología, teología, ética, religión, estudios culturales y lingüística.
        - **Ciencias Naturales, Matemáticas y Estadística:** comprende las ciencias biológicas como biología, bioquímica, genética y 
        biotecnología, así como ciencias ambientales, física, química, astronomía y matemáticas y estadística.
        - **Ciencias Sociales, Periodismo e Información:** abarca las ciencias sociales y del comportamiento como sociología, psicología, 
        antropología, ciencias políticas y economía. También incluye periodismo y bibliotecología y archivística.
        - **Educación:** abarca las teoría y práctica educativa, pedagogía, desarrollo curricular, evaluación educativa, y orientación y 
        consejería. También cubre la formación docente tanto general como especializada.
        - **Ingeniería, Industria y Construcción:** incluye todas las ramas de la ingeniería, tecnología de alimentos, textiles y construcción.
        - **Salud y Bienestar:** enfocado en la medicina, enfermería, odontología, farmacia, salud pública y fisioterapia.
        - **Servicios:** abarca servicios personales, gestión y operación de transportes y servicios de seguridad.
        - **Tecnologías de la Información y la Comunicación (TIC):** se centra en las ciencias de la computación, programación, redes, 
        ciberseguridad, desarrollo de software, y gestión de bases de datos.
        """)

    st.markdown(
        """
        En esta sección se omitió la categoría de programas y certificaciones genéricos, ya que son programas de educación que abarcan una amplia
        gama de temas y no se especializan en un campo específico. De igual forma se omitieron los programas no clasificados en ningún campo 
        amplio de educación.
        """)

with st.expander('🚢 **Exportaciones**'):

    st.markdown(
        """
        Se consultó el "Tejido Empresarial de Colombia" de ProColombia, el cual permite identificar las empresas con sede principal en el 
        municipio que realizaron al menos una exportación en los últimos 10 años (2013-2022). 
    """
    )

with st.expander('💵 **Inversión**'):

    st.markdown(
        """
        Se consultó el "Tejido Empresarial de Colombia" de ProColombia, el cual permite identificar las empresas con sede principal en el 
        municipio identificadas como sucursal de sociedad extranjera en la base del RUES o de Supersociedades. 
    """
    )

with st.expander('🛬 **Turismo**'):

    st.markdown(
        """
        Se consultó el "Tejido Empresarial de Colombia" de ProColombia, el cual permite identificar las empresas con sede principal en el 
        municipio relacionadas con actividades de turismo según su actividad económica principal. 
    """
    )

st.divider()

st.markdown("**En caso de tener alguna pregunta o comentario por favor contactar a:**")
st.markdown("Lina María Castro")
st.markdown("Asesor senior de Analítica")
st.markdown("lmcastro@procolombia.co")

st.divider()

st.subheader('Menú')

st.page_link("pages/1-Perfil.py", label="👆 :blue[**Perfiles municipios**]")
st.markdown("""Selecciona un municipio de Colombia y descubre toda la información disponible.""")
st.page_link("app.py", label="👆 :blue[**Página principal**]")
st.markdown("""Regresa a la página principal.""")