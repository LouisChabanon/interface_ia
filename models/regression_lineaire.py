from models.utils import Model, ModelType, separate_data
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression


class RegressionLineaire(Model):
    def __init__(self):
        super().__init__("Regression linéaire",
                         ModelType.REGRESSION, None)
        self.param = None
        self.name = "Regression linéaire"

    def run(self, data: pd.DataFrame, param: list):
        model = LinearRegression()
        separeted_data = separate_data(data, param[2])
        training_data = separeted_data["training_data"]
        predict_data = separeted_data["predict_data"]
        x_index, y_index = param[0], param[1]
        X, Y = training_data[x_index], training_data[y_index]
        reg = model.fit(X, Y)

        return reg.predict(predict_data[x_index])

    def display_parameters(self):
        st.write("Colonnes à utiliser pour la régression linéaire")
        x_index = st.text_input("Nom de la colonne x", "Heure")
        y_index = st.text_input("Nom de la colonne y", "Valeur")
        st.write("Part de l'échantillon pour l'entrainement")
        ratio = st.slider("Ratio", 0.1, 1.0, 0.8)

        return [x_index, y_index, ratio]

    def display_results(self):
        st.write(f"Résultat du modèle {self.name} avec paramètre {self.param}")
        return None
