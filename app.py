import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Listas de categorias (fora da função principal)
eye_colors = ['black', 'blue', 'brown', 'green', 'nesp', 'outros', 'red', 'white', 'yellow']
races = ['Android', 'Cyborg', 'God / Eternal', 'Human', 'Human / Radiation', 'Mutant', 'Symbiote', 'nesp', 'outros']
races_p = [ 'Cyborg', 'God / Eternal', 'Human', 'Human / Radiation', 'Mutant', 'Symbiote', 'nesp', 'outros']
hair_colors = ['auburn', 'black', 'blond', 'brown', 'green', 'nesp', 'no hair', 'outros', 'red', 'white']
publishers = ['DC Comics', 'Dark Horse Comics', 'George Lucas', 'Image Comics', 'Marvel Comics', 'NBC - Heroes', 'nesp', 'outros']

# Título da aplicação
st.title("Exploração de Dados de Super-Heróis e Modelos de Machine Learning")

# Carregando os dados
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Dados/Alelo/h_completo.csv")
    return df

# Carregando o modelo random forest
@st.cache_resource
def load_model_rf():
    with open('C:/Dados/Alelo/modelo_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_model_peso():
    with open('C:/Dados/Alelo/modelo_peso.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model_rf = load_model_rf()
model_peso = load_model_peso()
df = load_data()

# Criando abas
tab1, tab2, tab3, tab4 = st.tabs(["Exploração de Dados", "Análise de Clusters", "Previsão de Alinhamento", "Previsão do Peso"])

# Conteúdo da aba "Exploração de Dados"
with tab1:
    # Controles na barra lateral
    st.sidebar.header("Exploração de Dados")
    if st.sidebar.checkbox("Mostrar dados brutos"):
        st.subheader("Dados dos Super-Heróis")
        st.write(df.head())

    if st.sidebar.checkbox("Mostrar estatísticas descritivas"):
        st.subheader("Estatísticas Descritivas")
        st.write(df.describe())

    st.sidebar.header("Distribuição de Variáveis")
    variable = st.sidebar.selectbox("Selecione uma variável para visualizar a distribuição", df.columns)
    if st.sidebar.button("Plotar Distribuição"):
        st.subheader(f"Distribuição de {variable}")
        fig, ax = plt.subplots()
        sns.histplot(df[variable].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    st.sidebar.header("Filtrar Super-Heróis")
    alignment_options = ["Sem Filtro"] + list(df['Alignment'].unique())
    gender_options = ["Sem Filtro"] + list(df['Gender'].unique())
    publisher_options = ["Sem Filtro"] + list(df['Publisher'].unique())
    grupo_options = ["Sem Filtro"] + list(df['grupo'].unique())

    alignment_filter = st.sidebar.selectbox("Filtrar por Alinhamento", alignment_options)
    gender_filter = st.sidebar.selectbox("Filtrar por Gênero", gender_options)
    publisher_filter = st.sidebar.selectbox("Filtrar por Editora", publisher_options)
    grupo_filter = st.sidebar.selectbox("Filtrar por Grupo (Cluster)", grupo_options)

    filtered_data = df.copy()

    if alignment_filter != "Sem Filtro":
        filtered_data = filtered_data[filtered_data['Alignment'] == alignment_filter]
    if gender_filter != "Sem Filtro":
        filtered_data = filtered_data[filtered_data['Gender'] == gender_filter]
    if publisher_filter != "Sem Filtro":
        filtered_data = filtered_data[filtered_data['Publisher'] == publisher_filter]
    if grupo_filter != "Sem Filtro":
        filtered_data = filtered_data[filtered_data['grupo'] == grupo_filter]

    st.subheader("Super-Heróis Filtrados")
    st.write(filtered_data)

# Conteúdo da aba "Análise de Clusters"
with tab2:
    st.sidebar.header("Análise de Clusters")
    st.header("Análise de Clusters")
    field = st.sidebar.selectbox("Selecione um campo para análise", df.columns)

    def analyze_clusters(data, field, cluster_field="grupo"):
        # ... (seu código de análise de clusters) ...
        if data[field].dtype == "object":
            contingency_table = pd.crosstab(data[cluster_field], data[field])
            group_percentages = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

            st.subheader(f"Mapa de Calor: {field} por Grupo (N)")
            plt.figure(figsize=(10, 6))
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
            st.pyplot(plt)

            st.subheader(f"Mapa de Calor: {field} por Grupo (%)")
            plt.figure(figsize=(10, 6))
            sns.heatmap(group_percentages, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False)
            st.pyplot(plt)
        else:
            means = data.groupby(cluster_field)[field].mean()
            st.subheader(f"Médias de {field} por Grupo")
            st.write(means)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=means.index, y=means.values, palette="viridis")
            plt.xlabel("Grupo")
            plt.ylabel(f"Média de {field}")
            st.pyplot(plt)

    analyze_clusters(df, field)
    selected_cluster = st.sidebar.selectbox("Selecione um cluster para explorar", sorted(df["grupo"].unique()))
    cluster_data = df[df["grupo"] == selected_cluster]
    st.subheader(f"Super-Heróis no Cluster {selected_cluster}")
    st.write(cluster_data)

# Conteúdo da aba "Previsão de Alinhamento"
with tab3:
    st.sidebar.header("Previsão de Alinhamento")
    st.header("Previsão de Alinhamento")
    st.header("Entradas do Modelo")

    height = st.sidebar.number_input("Altura", value=170.0)
    weight = st.sidebar.number_input("Peso", value=70.0)
    total_powers = st.sidebar.number_input("Total de Poderes", value=1)
    gender = st.sidebar.selectbox("Gênero", ["Male", "Female", "nesp"])
    eye_color = st.sidebar.selectbox("Cor dos Olhos", eye_colors)
    race = st.sidebar.selectbox("Raça", races)
    hair_color = st.sidebar.selectbox("Cor do Cabelo", hair_colors)
    publisher = st.sidebar.selectbox("Editora", publishers)
    grupo = st.sidebar.selectbox("Grupo", [0, 1, 2, 3, 4, 5])

    data = {
        "Height": height,
        "Weight": weight,
        "total_de_poderes": total_powers,
        "Gender_rec_Female": gender == "Female",
        "Gender_rec_Male": gender == "Male",
        "Gender_rec_nesp": gender == "nesp",
    }

    for color in eye_colors:
        data[f"Eye color_rec_{color}"] = eye_color == color
    for r in races:
        data[f"Race_rec_{r}"] = race == r
    for h in hair_colors:
        data[f"Hair color_rec_{h}"] = hair_color == h
    for p in publishers:
        data[f"Publisher_rec_{p}"] = publisher == p
    for g in range(6):
        data[f"grupo_{g}"] = grupo == g

    input_data = pd.DataFrame([data])
    st.write("Dados de entrada:")
    st.write(input_data)

    if st.button("Prever Alinhamento"):
        prediction = model_rf.predict(input_data)
        if prediction[0] == 1:
            st.write("Alinhamento previsto: Bom")
        else:
            st.write("Alinhamento previsto: Mau")

# Conteúdo da aba "Previsão de Peso"
with tab4:
    st.sidebar.header("Previsão do Peso")
    st.header("Previsão do Peso")
    st.header("Entradas do Modelo")

    height_p = st.sidebar.number_input("Altura ", value=170.0)
    total_powers_p = st.sidebar.number_input("Total de Poderes ", value=1)
    gender_p = st.sidebar.selectbox("Gênero ", ["Male", "Female", "nesp"])
    race_p = st.sidebar.selectbox("Raça ", races)
    grupo_p = st.sidebar.selectbox("Grupo ", [0, 1, 2, 3, 4, 5])
    data_p = {
        "Height": height_p,
        "total_de_poderes": total_powers_p,
        "Gender_rec_Male": gender_p == "Male",
        "Gender_rec_nesp": gender_p == "nesp",
    }

    for r in races_p:
        data_p[f"Race_rec_{r}"] = race_p == r
    for g in range(1,6):
        data_p[f"grupo_{g}"] = grupo_p == g

    input_data_p = pd.DataFrame([data_p])
    st.write("Dados de entrada:")
    st.write(input_data_p)

    if st.button("Prever o Peso"):
        prediction_p = model_peso.predict(input_data_p)
        st.write(f"Peso previsto: {prediction_p[0]:.2f}") 
          
        
# Rodando a aplicação
#if __name__ == "__main__":
#    st.write("Aplicação carregada")