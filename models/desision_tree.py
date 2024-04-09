from models.utils import Model, ModelType, split_data
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd


class DecisionTree(Model):
    def __init__(self):
        self.param = None
        self.data = None
        self.name = "Decision Tree"
        self.type = ModelType.CLASSIFICATION
        super().__init__(self.name, self.type, self.param)

    def run(self):
        param = self.param
        data = split_data(self.data, param[2])
        algo = DecisionTreeClassifier(random_state=42)
        X = []
        for i in range(len(data["training_data"])):
            X.append([data["training_data"][param[0][0],
                     data["training_data"][param[0][1]]]]),
        y = data["training_data"][param[1]]
        result = algo.fit(X, y)
        return result.predict([data["predict_data"][param[0][0]], data["predict_data"][param[0][1]]])

    def display_parameters(self):
        data = self.data
        st.write("Colonnes à utiliser pour la classification KNN")
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

        x_index = (x_index_1, x_index_2)

        self.param = [x_index, y_index, ratio]

    def display_results(self, result):
        tree.plot_tree(result)
