from models.utils import Model, ModelType, split_data
import streamlit as st
import pandas as pd
import numpy as np


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class KMEAN(Model):
    def __init__(self):
        self.param = None
        self.data = None
        self.type = ModelType.CLASSIFICATION
        self.name = "KMean"
        super().__init__(self.name, self.type, self.param)


    def run(self):
        param = self.param
        self.data = split_data(self.data, param[2])
        model = KMeans(n_clusters=param[3], init='k-means++', n_init=10, max_iter=300,
                       tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
        training_data = self.data["training_data"]
        predict_data = self.data["predict_data"]
        x_index, y_index = param[0], param[1]
        x_data = np.column_stack(
            (training_data[x_index[0]], training_data[x_index[1]]))
        train = model.fit(x_data, training_data[y_index])
        x_data = np.column_stack(
            (predict_data[x_index[0]], predict_data[x_index[1]]))
        return train.predict(x_data)


    def display_parameters(self):
        data = self.data
        st.write("Colonnes à utiliser pour la classification KMean")
        x_index_1 = st.selectbox("Nom de la colonne du premier paramètre",
                                 list(data.columns), index=0)
        x_index_2 = st.selectbox("Nom de la colonne du second paramètre",
                                 list(data.columns), index=1)
        y_index = st.selectbox(
            "Nom de la colonne des labels", list(data.columns), index=2)

        if x_index_1 == y_index or x_index_2 == y_index or x_index_1 == x_index_2:
            st.error("Les colonnes doivent être différentes")
        st.write("Part de l'echantillon pour l'entrainement")
        ratio = st.slider("Ratio", 0.1, 1.0, 0.8)
        st.write("Nombre de clusters")
        k = st.slider("k", 1, 10, 5)

        x_index = (x_index_1, x_index_2)

        self.param = [x_index, y_index, ratio, k]

    # A revoir
    def display_results(self, result):
        data = self.data["predict_data"]
        param = self.param
        st.write("Résultats de la classification KMean")
        st.write(accuracy_score(data[param[1]], result))
        st.write("Visualisation des résultats")
        fig, ax = plt.subplots()
        ax.scatter(data[param[0][0]], data[param[0][1]], c=result)
        st.pyplot(fig)
