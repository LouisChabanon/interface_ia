from models.utils import Model, ModelType
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression


class RegressionLineaire(Model):
    def __init__(self, n_data):
        super().__init__("Regression linéaire",
                         ModelType.REGRESSION, None)
        self.param = None
        self.name = "Regression linéaire"
        self.data = n_data

    def run(self, data: dict[pd.DataFrame], param: dict):
        model = LinearRegression()
        training_data = data["training_data"]
        predict_data = data["predict_data"]
        x_index, y_index = param["x_index"], param["y_index"]
        X, Y = training_data[[x_index]], training_data[[y_index]]
        reg = model.fit(X, Y)

        return reg.predict(predict_data[["Heure"]])
    

    def display_parameters(self):
        st.table(self.data)


    def display_results(self):
        st.write(f"Résultat du modèle {self.name} avec paramètre {self.param}")
        return None
