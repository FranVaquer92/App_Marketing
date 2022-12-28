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
from PIL import Image


import streamlit as st
import SessionState

st.set_page_config(
     page_title="deepsense|Marketing",
     page_icon="ICONO DEEPSENSE.png",
     layout="wide",
     initial_sidebar_state="expanded",
     )

def app():
    #st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)
    st.image(Image.open('ICONO DEEPSENSE.png'), width = 100)
    st.title('Análisis avanzado de datos | Marketing')
    subido = 0


    with st.sidebar:
        st.image(Image.open('LOGO COLOR.png'), use_column_width=True)
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


    st.header('Cluster de segmentación de clientes')
    expander_pc = st.beta_expander("Extender", expanded=False)
    with expander_pc:
        df_clientes = pd.concat([df_2, clientes], axis = 1)
        df_clientes_group = df_clientes.groupby(by = ['CLIENTE'],as_index= False).mean().drop(["NUM_LINEA","MES", "AÑO", "CODIGO_PRODUCTO"], axis = 1)
        clientes_unicos = df_clientes_group['CLIENTE']
        df_clientes_group.drop('CLIENTE', axis = 1, inplace = True)

        scaler = StandardScaler()
        group_df_scaled = scaler.fit_transform(df_clientes_group)

        scores = []
        np.random.seed(seed = 100)

        range_values = range(1, 15)

        for i in range_values:
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(group_df_scaled)
            scores.append(kmeans.inertia_)
        plt.plot(range_values,scores, 'bx-')
        plt.title('Encontrar el número correcto de clusters')
        plt.xlabel('Nº Clusters')
        plt.ylabel('WCSS')
        st.pyplot()

        dif_scores = []


        for i in range(0,len(scores)):
            if i > 0:
                dif_scores.append(scores[i-1] - scores[i])

        n_clusters = np.argmax(dif_scores[1:])+2

        kmeans = KMeans(n_clusters)
        kmeans.fit(group_df_scaled)
        labels = kmeans.labels_

        st.subheader('Comportamiento de los diferentes clusters:')

        cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns = [df_clientes_group.columns])
        cluster_centers = scaler.inverse_transform(cluster_centers)
        cluster_centers = pd.DataFrame(data= cluster_centers, columns= [df_clientes_group.columns])
        st.dataframe(cluster_centers)

        y_kmeans = kmeans.fit_predict(group_df_scaled)

        st.subheader('A qué Cluster pertenece cada cliente:')
        clientes_group_cluster = pd.concat([df_clientes_group, pd.DataFrame({'cluster':labels}), clientes_unicos], axis = 1)
        clientes_group_cluster

        st.subheader('Distribución de los clusters')
        for i in df_clientes_group.columns[:4]:
            plt.figure(figsize=(30,n_clusters))
            for j in range(n_clusters):
                plt.subplot(1, n_clusters, j+1)
                cluster = clientes_group_cluster[clientes_group_cluster['cluster'] == j]
                cluster[i].hist()
                plt.title('{}  \nCluster - {}'.format(i,j))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        st.subheader('Reducción de la dimensinalidad utilizando PCA: Visualización de los clusters')     
        pca = PCA(n_components= 3)
        principal_comp = pca.fit_transform(group_df_scaled)

        pca_df = pd.DataFrame(data = principal_comp, columns=['pca1', 'pca2', 'pca3'])
        pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels}), clientes_unicos], axis = 1)
        fig = px.scatter_3d(pca_df, x = 'pca1', y = 'pca2', z= 'pca3',
                        color = 'cluster', symbol = 'cluster', hover_name = 'CLIENTE',size_max = 18, opacity = 0.7)
        fig.update_layout(margin = dict(l = 0, r = 0, t = 0))
        st.plotly_chart(fig) 

ss = SessionState.get(x=1)
if ss.x<=1:
    st.image(Image.open('LOGO COLOR.png'), width = 300)
titulo_object = st.empty()
password_object = st.empty()
inicio_object = st.empty()
if ss.x <= 1:
    pass_real = pd.read_csv("./password_usecases.csv")["PASSWORD"][0]
    titulo_object.title('Introduce contraseña')
    password = password_object.text_input("Contraseña")
    inicio_sesion = inicio_object.button("INICIAR SESION")
    if password == pass_real and inicio_sesion:
        ss.x = ss.x + 1
else:
    titulo_object.empty()
    password_object.empty()
    inicio_object.empty()
    
    #la funcion app es donde debe meterse el codigo de la aplicacion que queremos ejecutar
    app()
