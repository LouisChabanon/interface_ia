from models.utils import Model, ModelType
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class KNN(Model):
    def __init__(self, name: str, type: ModelType, parameters: list):
        super().__init__(name, type, parameters)
        self.param = None
        self.type = ModelType.CLASSIFICATION

    def run(self, data: dict[pd.DataFrame], param: dict):
        model = KNeighborsClassifier(n_neighbors=param["k"])

        training_data = data["training_data"]
        predict_data = data["predict_data"]
        x_index, y_index = param["x_index"], param["y_index"]
        X, Y = training_data[[x_index]], training_data[[y_index]]
        reg = model.fit(X, Y)

        return reg.predict(predict_data[[x_index]])
