import streamlit as st
import pandas as pd
import sklearn
import pickle
import joblib
import numpy as np

from f_extra import add_logo




# Puxando os dados e os modelos
df_filmes = st.session_state["df_filmes"]
df = st.session_state["df"]




# Corpo da pagina
st.write("# PROJETO TERA - CLASSIFICADOR DE FILMES! 🎬")


df_filmes
df