from utils import Model, ModelType
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression


class RegressionLineaire(Model):
    def __init__(self):
        super().__init__("Regression linéaire",
                         ModelType.REGRESSION, None)
        self.param = None

    def run(self, data: pd.DataFrame, param: dict):
        model = LinearRegression()
        x, y = data.iloc[:, :-1], data.iloc[:, -1]
        reg = model.fit(x, y)
        return reg

    def display_results(self):
        return None
