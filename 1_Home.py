import streamlit as st
import pandas as pd
import sklearn
import pickle
import joblib
import numpy as np

from f_extra import add_logo

# Configurações da página
st.set_page_config(
    page_title="Home",
    page_icon="📽",
    layout="wide"
)

# Funções para pegar os dados e os modelos
@st.cache_data
def get_dados():
    return pd.read_csv(r"bases_de_dados/df_filmes_pre_processada.csv", index_col=0, sep=";")

@st.cache_data
def get_models():
    lr_clf = pickle.load(open(r'models/classificacao/lr_clf.pkl', 'rb'))
    rf_clf = pickle.load(open(r'models/classificacao/Rf_clf.pkl', 'rb'))
    xgb_clf = pickle.load(open(r'models/classificacao/Xgb_clf.pkl', 'rb'))
    svm_clf = pickle.load(open(r'models/classificacao/svm_clf.pkl', 'rb'))
    lr_reg = pickle.load(open(r'models/regressao/lr_reg.pkl', 'rb'))
    rf_reg = pickle.load(open(r'models/regressao/Rf_reg.pkl', 'rb'))
    xgb_reg = pickle.load(open(r'models/regressao/Xgb_reg.pkl', 'rb'))
    svm_reg = pickle.load(open(r'models/regressao/svm_reg.pkl', 'rb'))

    return lr_clf, rf_clf, xgb_clf, svm_clf, lr_reg, rf_reg, xgb_reg, svm_reg

@st.cache_data
def get_encoder():
    OHE_CLF = joblib.load(r"encoders/one_hot_encoder_com_diretor.pkl")
    OHE_REG = joblib.load(r"encoders/one_hot_encoder_com_diretor_revenue.pkl")
    return  OHE_CLF, OHE_REG




@st.cache_data
def get_scaler():
    SCALER_CLF = joblib.load(r"scalers/standard_scaler_com_diretor.pkl")
    SCALER_REG = joblib.load(r"scalers/standard_scaler_com_diretor.pkl")
    return SCALER_CLF, SCALER_REG


# Puxando os dados e os modelos
df_filmes = get_dados()
lr_clf, rf_clf, xgb_clf, svm_clf, lr_reg, rf_reg, xgb_reg, svm_reg = get_models()
OHE_CLF, OHE_REG = get_encoder()
SCALER_CLF, SCALER_REG =  get_scaler()


# Crie as listas
lista_tipos_script = df_filmes['Script Type'].unique().tolist()
lista_diretores = df_filmes['Director'].unique().tolist()

# Session State
st.session_state["df_filmes"] = df_filmes
st.session_state["lista_tipos_script"] = lista_tipos_script
st.session_state["lista_diretores"] = lista_diretores
st.session_state["OHE_CLF"] = OHE_CLF
st.session_state["OHE_REG"] = OHE_REG
st.session_state["SCALER_CLF"] = SCALER_CLF
st.session_state["SCALER_REG"] = SCALER_REG
st.session_state["lr_clf"] = lr_clf
st.session_state["rf_clf"] = rf_clf
st.session_state["xgb_clf"] = xgb_clf
st.session_state["svm_clf"] = svm_clf
st.session_state["lr_reg"] = lr_reg
st.session_state["rf_reg"] = rf_reg
st.session_state["xgb_reg"] = xgb_reg
st.session_state["svm_reg"] = svm_reg


# Corpo da pagina
st.write("# PROJETO TERA - CLASSIFICADOR DE FILMES! 🎬")

df_filmes

st.sidebar.markdown("Feito por [Ana Carolina Novaes Silva](https://www.linkedin.com/in/ana-carolina-novaes-silva-723a22b9/)")
st.sidebar.markdown("Feito por [Ciro Menescal da Silva Campos](https://www.linkedin.com/in/ciro-menescal-da-silva-campos-396462179/)")
st.sidebar.markdown("Feito por [Diego Jorge Maia De Sousa](https://www.linkedin.com/in/diegojmsousa/)")
st.sidebar.markdown("Feito por [Filipe Casal](https://www.linkedin.com/in/filipecasalf/)")
st.sidebar.markdown("Feito por [Iany]()")
st.sidebar.markdown("Feito por [Leonardo Shimizu](https://www.linkedin.com/in/leonardo-shimizu-09715a92/)")
st.sidebar.markdown("Feito por [Wolney Barreto Garção Filho](https://www.linkedin.com/in/wolneybarreto/)")

st.markdown(
    """
## Bem-vindo ao Nosso Projeto de Previsão de Sucesso de Filmes

Nossa missão é usar a inteligência artificial para prever o sucesso financeiro de filmes. Usamos um conjunto de dados abrangente e uma variedade de modelos de Machine Learning, incluindo Regressão Logística, Decision Trees, XGBoost e SVM (Support Vector Machine), para determinar se um filme será um sucesso de bilheteria.

Se você está interessado em descobrir o que torna um filme um sucesso nas bilheteiras, você está no lugar certo. Nossa equipe está comprometida em fornecer informações valiosas para produtores de cinema e amantes da sétima arte.

Explore nosso site para saber mais sobre como nossos modelos funcionam e como podemos ajudá-lo a tomar decisões mais informadas na indústria cinematográfica. Juntos, estamos moldando o futuro do cinema com a ajuda da inteligência artificial.
    """
)