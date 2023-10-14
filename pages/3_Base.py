import streamlit as st
import pandas as pd
import sklearn
import pickle
import joblib
import numpy as np
from f_extra import add_logo

# Funções para pegar os dados e os modelos
@st.cache_data
def get_dados():
    return pd.read_csv(r"bases_de_dados/df_filmes_pre_processada.csv", index_col=0, sep=";")

# Puxando os dados e os modelos
df_filmes = get_dados()

df_filmes