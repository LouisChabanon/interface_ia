from models.utils import Model, ModelType
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


class Randomforest(Model):
    def __init__(self):
        self.param = None
        self.type = ModelType.CLASSIFICATION
        self.name = "Randomforest"
        super().__init__(self.name, self.type, self.param)

    def run(self, data: list[pd.DataFrame], param: dict):
        model = RandomForestRegressor(n_estimators=param[3], random_state=0)
        training_data = data["training_data"]
        predict_data = data["predict_data"]
        x_index, y_index = param[0], param[1]
        x_data = np.column_stack(
            (training_data[x_index[0]], training_data[x_index[1]]))
        train = model.fit(x_data, training_data[y_index])
        x_data = np.column_stack(
            (predict_data[x_index[0]], predict_data[x_index[1]]))
        return train.predict(x_data)

    def display_parameters(self, data: pd.DataFrame):
        st.write("Colonnes à utiliser pour la classification Randomforest")
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
        st.write("Nombre d'estimateurs")
        k = st.slider("k", 1, 10, 5)

        x_index = (x_index_1, x_index_2)

        return [x_index, y_index, ratio, k]

    # A revoir
    def display_results(self, data: pd.DataFrame, result, param: list):
        st.write("Résultats de la classification Randomforest")
        st.write(accuracy_score(data[param[1]], result))
        st.write("Visualisation des résultats")
        fig, ax = plt.subplots()
        ax.scatter(data[param[0][0]], data[param[0][1]], c=result)
        st.pyplot(fig)
