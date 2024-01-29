import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import plotly.express as px
import plotly.graph_objs as go
import io


def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def knn_imputer(data):
    num_data = data.select_dtypes(include=np.number)
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(num_data)
    imputed_df = pd.DataFrame(imputed_data, columns=num_data.columns)
    return imputed_df

def create_plots(data, columns, plot_type):
    plot_functions = {
        "Histograma": lambda: px.histogram(data.melt(id_vars=columns, var_name='variable'), nbins=30, x='value', color='variable', facet_col='variable', facet_col_wrap=2),
        "Diagrama de caja": lambda: go.Figure([go.Box(y=data[col], name=col) for col in columns]),
        "Diagrama de violín": lambda: go.Figure([go.Violin(y=data[col], name=col, box_visible=True) for col in columns]),
        "Gráfico de dispersión": lambda: px.scatter_matrix(data, dimensions=columns),
        "Gráfico de barras": lambda: px.bar(data, x=data.index, y=columns),
        "Gráfico de línea": lambda: px.line(data, x=data.index, y=columns),
    }

    fig = plot_functions[plot_type]()

    return fig


st.title("Análisis y visualización de datos CSV")

uploaded_file = st.file_uploader("Carga tu archivo CSV", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Datos originales:")
    st.write(data.head())

    imputed_data = knn_imputer(data)
    st.write("Datos con valores nulos reparados usando KNN Imputer:")
    st.write(imputed_data.head())

    st.write("Información sobre el DataFrame:")
    buf = io.StringIO()
    imputed_data.info(buf=buf)
    st.write(buf.getvalue())

    columns = st.multiselect("Selecciona las columnas para graficar", imputed_data.columns)
    plot_types = st.multiselect("Elige los tipos de gráfico", ["Histograma", "Diagrama de caja", "Diagrama de violín", "Gráfico de dispersión", "Gráfico de barras", "Gráfico de línea"])

    if len(columns) > 0 and len(plot_types) > 0:
        for plot_type in plot_types:
            fig = create_plots(imputed_data, columns, plot_type)
            st.plotly_chart(fig)

    st.write("Descripción de las variables numéricas:")
    st.write(imputed_data.describe())

    st.write("Descripción de las variables categóricas:")
    st.write(data.select_dtypes(include="object").describe())

    st.write("Matriz de correlación de las variables numéricas:")
    corr_matrix = imputed_data.corr()
    st.write(corr_matrix)

    st.write("Mapa de calor de la matriz de correlación:")
    fig_corr = px.imshow(corr_matrix, x=imputed_data.columns, y=imputed_data.columns)
    st.plotly_chart(fig_corr)
