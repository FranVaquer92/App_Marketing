import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import plotly.express as px
from IPython.display import display
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import random

import streamlit as st

st.set_page_config(
     page_title="Deepsense|Marketing",
     page_icon="chart_with_upwards_trend",
     layout="wide",
     initial_sidebar_state="expanded",
     )

st.title('Segmentación y clasificación de clientes.')
subido = 0


with st.sidebar:
    st.subheader('Sube tu dataset')
    data_file = st.file_uploader("Sube el archivo CSV",type=['csv'])
    if data_file is not None:
        subido = 1
    if st.button("Process"):
        if data_file is not None:
            file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
            st.write(file_details)

if subido == 1:
    st.subheader('Este es el dataset que has subido')
else:
    st.subheader('Este es el dataset de ejemplo')
"""
#### Recuerda que el dataset debe de seguir la siguiente estructura:
"""

"""
CANTIDAD | PRECIO UNITARIO | NÚMERO DE LÍNEA O PEDIDO | VENTA | FECHA | MES | AÑO | PRODUCTO | MSRP | CÓDIGO PRODUCTO | NOMBRE CLIENTE | PAÍS | OFERTA |
"""
try:
    encoding = "unicode_escape"
    df = pd.read_csv(data_file, encoding=encoding)
#    st.dataframe(df)
except:
    try:
        encoding = "utf-8"
        df = pd.read_csv(data_file, encoding=encoding)
#        st.dataframe(df)
    except:
        df = pd.read_csv('sales_data_summary.csv')
#        st.dataframe(df)
def barplotvisualization(x, posicion):
    fig = plt.Figure(figsize=(20,20))
    fig = px.bar(x = df[x].value_counts().index, 
                 y = df[x].value_counts(), 
                 color = df[x].value_counts().index, 
                 height = 400)
    if posicion == 'col1':
        col1.plotly_chart(fig, use_column_width=True)
    elif posicion == 'col2':
        col2.plotly_chart(fig, use_column_width=True)
    
df.columns = ["CANTIDAD", "PRECIO_UNITARIO", "NUM_LINEA", "VENTA", "FECHA", "MES", "AÑO", "PRODUCTO", "MSRP", "CODIGO_PRODUCTO", "CLIENTE", "PAIS", "OFERTA"]


st.dataframe(df)


col1, col2 = st.beta_columns((3, 1))

col1.subheader('Clientes')
barplotvisualization('CLIENTE', 'col1')

col2.subheader('Productos')
barplotvisualization('PRODUCTO', 'col2')

col1.subheader('País')
barplotvisualization('PAIS', 'col1')

df_group = df.groupby(by = 'FECHA').sum()
fig = px.line(x = df_group.index, y = df_group.VENTA, title='EVOLUCIÓN DE LAS VENTAS')
col2.plotly_chart(fig)

plt.figure(figsize= (10,10))

clientes = df['CLIENTE']

df.drop('CLIENTE', axis = 1, inplace=True)

j = 0
for i in range(8):
    if df.columns[i]!= 'NUM_LINEA' and df.columns[i]!= 'FECHA' and df.columns[i]!= 'PRODUCTO' and df.columns[i]!= 'OFERTA' and df.columns[i]!= 'AÑO':
        if j%2 == 0:
            fig = ff.create_distplot([df[df.columns[i]].apply(lambda x: float(x))], ['displot'])
            fig.update_layout(title_text = df.columns[i])
            col1.plotly_chart(fig)
        else:
            fig = ff.create_distplot([df[df.columns[i]].apply(lambda x: float(x))], ['displot'])
            fig.update_layout(title_text = df.columns[i])
            col2.plotly_chart(fig)
        j+=1
        

st.header('Cluster de patrones de compra')
expander_pc = st.beta_expander("Extender", expanded=False)
with expander_pc:
    st.subheader('Distribución de los clusters')

    def dummies(x):
        dummy = pd.get_dummies(df[x])
        df.drop(columns = x, inplace= True)
        return pd.concat([df, dummy], axis = 1)

    df = dummies('PAIS')
    df = dummies('PRODUCTO')
    df = dummies('OFERTA')

    df_2 = df.drop('FECHA', axis = 1, inplace= False)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_2)

    kmeans = KMeans(3)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_

    st.subheader('Patrones de compra diferenciados:')

    cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns = [df_2.columns])
    cluster_centers = scaler.inverse_transform(cluster_centers)
    cluster_centers = pd.DataFrame(data= cluster_centers, columns= [df_2.columns])
    st.dataframe(cluster_centers)

    y_kmeans = kmeans.fit_predict(df_scaled)
    df_cluster = pd.concat([df_2, pd.DataFrame({'cluster':labels})], axis = 1)


    df_2['NUM_LINEA'] = df_2['NUM_LINEA'].apply(lambda x: float(x))

    n_clusters = labels.max() +1
    for i in df_2.columns[:8]:
        plt.figure(figsize=(30,n_clusters))
        for j in range(n_clusters):
            plt.subplot(1, n_clusters, j+1)
            cluster = df_cluster[df_cluster['cluster'] == j]
            cluster[i].hist()
            plt.title('{}  \nCluster - {}'.format(i,j))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    st.subheader('Reducción de la dimensinalidad utilizando PCA: Visualización de los clusters')    

    pca = PCA(n_components= 3)
    principal_comp = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data = principal_comp, columns=['pca1', 'pca2', 'pca3'])
    pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels}),clientes], axis = 1)

    fig = px.scatter_3d(pca_df, x = 'pca1', y = 'pca2', z= 'pca3',
                        color = 'cluster', 
                        size_max = 18, opacity = 0.7)
    fig.update_layout(margin = dict(l = 0, r = 0, t = 0))
    st.plotly_chart(fig)    



