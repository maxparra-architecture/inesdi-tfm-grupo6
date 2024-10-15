import streamlit as st
import os

# Rutas
path_cwd = os.getcwd()
path_data = path_cwd + "/data/"
path_img  = path_cwd + "/img/"

def menu():
    # Logo ProColombia
    st.sidebar.image(path_img + "Logo_ProColombia.png", use_column_width=True)
    st.sidebar.divider()
    st.sidebar.text("")
    st.sidebar.page_link("app.py", label="▶ :gray[Página Principal]")
    st.sidebar.page_link("pages/1-Perfil.py", label="▶ :gray[Perfiles municipios]")
    st.sidebar.page_link("pages/2-Metodologia.py", label="▶ :gray[Metodología]")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.subheader("Elaborado por:") 
    st.sidebar.markdown("####  Coordinación de Analítica, Gerencia de Inteligencia Comercial, ProColombia.")
    st.sidebar.divider()
    # Logo Ministerio
    st.sidebar.image(path_img + "Logo_MinCit.png", use_column_width=True)