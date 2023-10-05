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

# Calcular o valor máximo com base na coluna "Budget ($million)" do DataFrame
valor_maximo_orcamento = df_filmes["Budget ($million)"].max()
valor_maximo_orcamento = float(valor_maximo_orcamento)

# Calcular o valor máximo com base na coluna "Runtime (Minutes)" do DataFrame
valor_maximo_runtime = df_filmes["Runtime (Minutes)"].max()
valor_maximo_runtime = float(valor_maximo_runtime)

# Corpo da pagina
st.write("# PROJETO TERA - CLASSIFICADOR DE FILMES! 🎬")

df_filmes

st.write("# Crie o seu filmes! 🎬")
diretor = st.selectbox("Nome do Diretor", lista_diretores)
tipo_script = st.selectbox("Tipo de Script", lista_tipos_script)
orcamento = st.slider("Orçamento ($million)", min_value=0.0, max_value=valor_maximo_orcamento, value=0.0, step=1.0)
runtime = st.slider("Duração do Filme (Minutes)", min_value=0.0, max_value=valor_maximo_runtime, value=0.0, step=1.0)
categoria_action = st.checkbox("Action")
categoria_adventure = st.checkbox("Adventure")
categoria_animation = st.checkbox("Animation")
categoria_biography = st.checkbox("Biography")
categoria_comedy = st.checkbox("Comedy")
categoria_crime = st.checkbox("Crime")
categoria_drama = st.checkbox("Drama")
categoria_family = st.checkbox("Family")
categoria_fantasy = st.checkbox("Fantasy")
categoria_history = st.checkbox("History")
categoria_horror = st.checkbox("Horror")
categoria_music = st.checkbox("Music")
categoria_musical = st.checkbox("Musical")
categoria_mystery = st.checkbox("Mystery")
categoria_romance = st.checkbox("Romance")
categoria_sci_fi = st.checkbox("Sci-Fi")
categoria_sport = st.checkbox("Sport")
categoria_thriller = st.checkbox("Thriller")
categoria_war = st.checkbox("War")
categoria_western = st.checkbox("Western")

# Suponha que você tenha um DataFrame df_filmes com a coluna "Oscar Winners" contendo valores booleanos (True/False)

# Definir uma lista de categorias
categorias = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]

# Criar um dicionário para armazenar os valores selecionados pelo usuário
dados = {
    "Director": [diretor],
    "Script Type": [tipo_script],
    "Budget ($million)": [orcamento],
    "Oscar Winners": [df_filmes[df_filmes["Director"] == diretor]["Oscar Winners"].values[0]],  # Obtém o valor de "Oscar Winners" do diretor selecionado
    "Runtime (Minutes)": [runtime]
}

# Adicionar as categorias ao dicionário
for categoria in categorias:
    valor_categoria = locals().get(f"categoria_{categoria.lower().replace(' ', '_')}", False)  # Obter o valor da variável da categoria
    dados[categoria] = [1 if valor_categoria else 0]  # 1 se estiver marcado, 0 caso contrário

# Criar um DataFrame com os dados
df = pd.DataFrame(dados)

# Exibir o DataFrame
st.write(df)

# Defina as colunas categóricas que deseja codificar
colunas_categoricas = ['Script Type', 'Director']

# Transforme as colunas categóricas em colunas one-hot usando o codificador treinado
categorias_encoded = OHE_CLF.transform(df[colunas_categoricas])

# Crie um novo DataFrame com as colunas one-hot codificadas
categorias_encoded_df = pd.DataFrame(categorias_encoded, columns=OHE_CLF.get_feature_names_out(colunas_categoricas))

# Concatene 'categorias_encoded_df' ao DataFrame original para incluir as colunas one-hot
df = pd.concat([df, categorias_encoded_df], axis=1)

# Remova as colunas originais 'Script Type' e 'Director', se desejar
df.drop(colunas_categoricas, axis=1, inplace=True)

df['Budget ($million)'] = df['Budget ($million)'].astype(float)

df_CLF = SCALER_CLF.transform(df)
df_REG = SCALER_REG.transform(df)

drop_col = [374,375,376]
# df_CLF = np.delete(df_CLF, drop_col, axis=1)
df_REG = np.delete(df_REG, drop_col, axis=1)

# Modelos de classificação
resultado_classificacao_lr = lr_clf.predict(df_CLF)
resultado_classificacao_rf = rf_clf.predict(df_CLF)
resultado_classificacao_xgb = xgb_clf.predict(df_CLF)
resultado_classificacao_svm = svm_clf.predict(df_CLF)

# Modelos de regressão
resultado_regressao_lr = lr_reg.predict(df_REG)
resultado_regressao_rf = rf_reg.predict(df_REG)
resultado_regressao_xgb = xgb_reg.predict(df_REG)
resultado_regressao_svm = svm_reg.predict(df_REG)

# Corpo do modelo

# Suponha que você tenha resultados de modelos para cada modelo
resultados_modelos = {
    "Regressão Linear e Logistica [Regressão Quebrada]": {
        "Classificação (Sucesso)": resultado_classificacao_lr,  # Substitua com o resultado real
        "Regressão (Valor)": resultado_regressao_lr,  # Substitua com o resultado real
    },
    "Randon Forest": {
        "Classificação (Sucesso)": resultado_classificacao_rf,  # Substitua com o resultado real
        "Regressão (Valor)": resultado_regressao_rf,  # Substitua com o resultado real
    },
    "SVM": {
        "Classificação (Sucesso)": resultado_classificacao_svm,  # Substitua com o resultado real
        "Regressão (Valor)": resultado_regressao_svm,  # Substitua com o resultado real
    },
    "Gradient Boosting": {
        "Classificação (Sucesso)": resultado_classificacao_xgb,  # Substitua com o resultado real
        "Regressão (Valor)": resultado_regressao_xgb,  # Substitua com o resultado real
    }
}

# Criar o layout com duas colunas para cada modelo
st.write("# Resultados dos Modelos")

# Loop através dos modelos
for nome_modelo, resultados in resultados_modelos.items():
    st.write(f"## {nome_modelo}")
    
    # Coluna 1: Classificação (Sucesso ou Não)
    st.write("### Classificação (Sucesso ou Não)")
    resultado_classificacao = resultados["Classificação (Sucesso)"]
    if resultado_classificacao == 1:
        st.write("O filme provavelmente fará sucesso.")
    else:
        st.write("O filme provavelmente não fará sucesso.")
    
    # Coluna 2: Regressão (Valor)
    st.write("### Arrecadação da Primeira Semana: Regressão (Valor)")
    resultado_regressao = resultados["Regressão (Valor)"]
    st.write(f"O valor estimado é: {resultado_regressao} milhões de dólares")
    
    # Separador entre modelos
    st.write("---")







