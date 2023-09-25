import streamlit as st
import pandas as pd
import pickle




# Puxando os dados e os modelos
df_filmes = st.session_state["df_filmes"]




# Corpo da pagina
st.write("# PROJETO TERA - CLASSIFICADOR DE FILMES! 🎬")


df_filmes