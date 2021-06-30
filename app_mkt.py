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

random.seed(100)

st.header('Segmentación y clasificación de clientes.')
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
CANTIDAD | PRECIO UNITARIO | NÚMERO DE LÍNEA O PEDIDO | VENTA | FECHA | MES | AÑO | PRODUCTO | MSRP | CÓDIGO PRODUCTO | PAÍS | OFERTA |
"""
try:
    encoding = "unicode_escape"
    df = pd.read_csv(data_file, encoding=encoding)
    st.dataframe(df)
except:
    try:
        encoding = "utf-8"
        df = pd.read_csv(data_file, encoding=encoding)
        st.dataframe(df)
    except:
        df = pd.read_csv('sales_data_summary.csv')
        st.dataframe(df)
def barplotvisualization(x):
    fig = plt.Figure(figsize=(20,20))
    fig = px.bar(x = df[x].value_counts().index, 
                 y = df[x].value_counts(), 
                 color = df[x].value_counts().index, 
                 height = 600)
    st.plotly_chart(fig)

df.columns = ["CANTIDAD", "PRECIO_UNITARIO", "NUM_LINEA", "VENTA", "FECHA", "MES", "AÑO", "PRODUCTO", "MSRP", "CODIGO_PRODUCTO", "PAIS", "OFERTA"]


st.subheader('Productos')
barplotvisualization('PRODUCTO')

st.subheader('País')
barplotvisualization('PAIS')

df_group = df.groupby(by = 'FECHA').sum()
fig = px.line(x = df_group.index, y = df_group.VENTA, title='EVOLUCIÓN DE LAS VENTAS')
st.plotly_chart(fig)

plt.figure(figsize= (10,10))

st.subheader('Distribución según las diferentes variables')

for i in range(8):
    if df.columns[i]!= 'NUM_LINEA' and df.columns[i]!= 'FECHA' and df.columns[i]!= 'PRODUCTO' and df.columns[i]!= 'OFERTA' and df.columns[i]!= 'AÑO':
        fig = ff.create_distplot([df[df.columns[i]].apply(lambda x: float(x))], ['displot'])
        fig.update_layout(title_text = df.columns[i])
        st.plotly_chart(fig)
        
plt.figure(figsize=(15,15))

fig = px.scatter_matrix(df, dimensions=df.drop('FECHA', axis = 1, inplace= False).columns, color = 'MES')

fig.update_layout(
    title = 'DISTRIBUCIÓN MENSUAL',
    width = 1100,
    height = 1100
)

st.plotly_chart(fig)


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

cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns = [df_2.columns])
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data= cluster_centers, columns= [df_2.columns])
y_kmeans = kmeans.fit_predict(df_scaled)
df_cluster = pd.concat([df_2, pd.DataFrame({'cluster':labels})], axis = 1)


df_2['NUM_LINEA'] = df_2['NUM_LINEA'].apply(lambda x: float(x))

st.subheader('Distribución de los clusters')

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

st.subheader('Reducción de la dimensinalidad utilizando PCA: Visualización de los clusters de clientes')    

pca = PCA(n_components= 3)
principal_comp = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(data = principal_comp, columns=['pca1', 'pca2', 'pca3'])
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)

fig = px.scatter_3d(pca_df, x = 'pca1', y = 'pca2', z= 'pca3',
                    color = 'cluster', 
                    size_max = 18, opacity = 0.7)
fig.update_layout(margin = dict(l = 0, r = 0, t = 0))
st.plotly_chart(fig)    

