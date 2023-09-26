import streamlit as st
import pandas as pd
import pickle
import joblib

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
    return pd.read_excel("bases de dado\df_filmes_pre_processada.xlsx", index_col=0)

@st.cache_data
def get_models():
    lr_clf = pickle.load(open('models\classificacao\lr_clf.pkl', 'rb'))
    rf_clf = pickle.load(open('models\classificacao\Rf_clf.pkl', 'rb'))
    xgb_clf = pickle.load(open('models\classificacao\Xgb_clf.pkl', 'rb'))
    svm_clf = pickle.load(open('models\classificacao\svm_clf.pkl', 'rb'))
    lr_reg = pickle.load(open('models\regressao\lr_reg.pkl', 'rb'))
    rf_reg = pickle.load(open('models\regressao\Rf_reg.pkl', 'rb'))
    xgb_reg = pickle.load(open('models\regressao\Xgb_reg.pkl', 'rb'))
    svm_reg = pickle.load(open('models\regressao\svm_reg.pkl', 'rb'))

    return lr_clf, rf_clf, xgb_clf, svm_clf, lr_reg, rf_reg, xgb_reg, svm_reg

@st.cache_data
def get_encoder():
    OHE_CLF = joblib.load(r"encoders\one_hot_encoder_com_diretor.pkl")
    OHE_REG = joblib.load(r"encoders\one_hot_encoder_com_diretor_revenue.pkl")
    return  OHE_CLF, OHE_REG

@st.cache_data
def get_scaler():
    SCALER_CLF = joblib.load(r"scalers\standard_scaler_com_diretor.pkl")
    SCALER_REG = joblib.load(r"scalers\standard_scaler_com_diretor.pkl")
    return SCALER_CLF, SCALER_REG


# Puxando os dados e os modelos
df_filmes = get_dados()
st.session_state["df_filmes"] = df_filmes
lr_clf, rf_clf, xgb_clf, svm_clf, lr_reg, rf_reg, xgb_reg, svm_reg = get_models()
OHE_CLF, OHE_REG = get_encoder()
SCALER_REG, SCALER_REG =  get_scaler()


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