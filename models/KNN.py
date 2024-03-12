from models.utils import Model, ModelType
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class KNN(Model):
    def __init__(self, name: str, type: ModelType, parameters: list):
        super().__init__(name, type, parameters)
        self.param = None
        self.type = ModelType.CLASSIFICATION

    def run(self, data: list[pd.DataFrame], param: dict):
        model = KNeighborsClassifier(n_neighbors=param[2])
        training_data = data["training_data"]
        predict_data = data["predict_data"]
        x_index, y_index = param[0], param[1]
        train = model.fit(training_data[[x_index]], training_data[y_index])
        return train.predict(predict_data[[x_index]])
    
        
    def display_parameters(self, data: pd.DataFrame):
        st.write("Colonnes à utiliser pour la classification KNN")
        x_index = st.selectbox("Nom de la colonne x", list(data.columns), index=0)
        y_index = st.selectbox("Nom de la colonne y", list(data.columns), index=1)
        if x_index == y_index:
            st.error("Les colonnes x et y doivent être différentes")
        st.write("Nombre de voisins")
        k = st.slider("k", 1, 10, 5)

        return [x_index, y_index, k, ]
