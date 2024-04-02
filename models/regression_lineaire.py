from models.utils import Model, ModelType, separate_data
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class RegressionLineaire(Model):
    def __init__(self):
        super().__init__("Regression linéaire", ModelType.REGRESSION, None)
        self.param = None
        self.data = None
        self.name = "Regression linéaire"

    def run(self):
        model = LinearRegression()
        self.data = separate_data(self.data, self.param[2])
        training_data = self.data["training_data"]
        predict_data = self.data["predict_data"]
        x_index, y_index = self.param[0], self.param[1]
        X, Y = training_data[[x_index]], training_data[[y_index]]
        reg = model.fit(X, Y)

        return reg.predict(predict_data[[x_index]])

    def display_parameters(self, data: pd.DataFrame):
        st.write("Colonnes à utiliser pour la régression linéaire")
        x_index = st.selectbox("Nom de la colonne x",
                               list(data.columns), index=0)
        y_index = st.selectbox("Nom de la colonne y",
                               list(data.columns), index=1)
        if x_index == y_index:
            st.error("Les colonnes x et y doivent être différentes")
        st.write("Part de l'échantillon pour l'entrainement")
        ratio = st.slider("Ratio", 0.1, 1.0, 0.8)
        return [x_index, y_index, ratio]

    def display_results(self,result):
        st.write("Résultats de la régression linéaire")
        fig, ax = plt.subplots()
        ax.scatter(self.data[self.param[0]], self.data[self.param[1]])
        ax.plot(self.data[self.param[0]], result, color="red")
        st.pyplot(fig)
