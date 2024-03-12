from models.utils import Model, ModelType, separate_data
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class RegressionLineaire(Model):
    def __init__(self):
        super().__init__("Regression linéaire",
                         ModelType.REGRESSION, None)
        self.param = None
        self.name = "Regression linéaire"

    def run(self, separated_data: pd.DataFrame, param: list):
        model = LinearRegression()
        training_data = separated_data["training_data"]
        predict_data = separated_data["predict_data"]
        x_index, y_index = param[0], param[1]
        X, Y = training_data[[x_index]], training_data[[y_index]]
        reg = model.fit(X, Y)


        return reg.predict(predict_data[[x_index]])


    def display_parameters(self, data: pd.DataFrame):
        st.write("Colonnes à utiliser pour la régression linéaire")
        x_index = st.selectbox("Nom de la colonne x", list(data.columns), index=0)
        y_index = st.selectbox("Nom de la colonne y", list(data.columns), index=1)
        if x_index == y_index:
            st.error("Les colonnes x et y doivent être différentes")
        st.write("Part de l'échantillon pour l'entrainement")
        ratio = st.slider("Ratio", 0.1, 1.0, 0.8)

        return [x_index, y_index, ratio]

    def display_results(self, data: pd.DataFrame, result, param: list):
        st.write("Résultats de la régression linéaire")

        fig, ax = plt.subplots()
        ax.scatter(data[param[0]], data[param[1]])
        ax.plot(data[param[0]], result, color='red')
        st.pyplot(fig)
