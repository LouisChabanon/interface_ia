from models.utils import Model, ModelType, split_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RegressionPolynomiale(Model):
    def __init__(self):
        self.param = None
        self.type = ModelType.REGRESSION
        self.name = "Regression Polynomiale"
        super().__init__(self.name, self.type, self.param)

    def display_parameters(self):
        data = self.data
        st.write("Colonnes à utiliser pour la régression polynomiale")
        x_index = st.selectbox("Nom de la colonne des x",
                               list(data.columns), index=0)
        y_index = st.selectbox("Nom de la colonne des y",
                               list(data.columns), index=1)
        
        st.write("Degré du polynome")
        degree = st.slider("Degré", 1, 10, 2)

        self.param = [x_index, y_index, degree]

    def run(self):
        feature = PolynomialFeatures(degree=self.param[2])
        X = feature.fit_transform(self.data[[self.param[0]]])
        model = make_pipeline(PolynomialFeatures(self.param[2]), LinearRegression())
        model.fit(X, self.data[self.param[1]])
        
        X_new = self.data[[self.param[0]]].reshape(-1, 1)