from utils import Model, ModelType
import streamlit as st


class RegressionLineaire(Model):
    def __init__(self):
        super().__init__("Regression linéaire",
                         ModelType.REGRESSION, None)
        self.param = 0

    def display_parameters(self):
        self.param = st.slider("Choisir le paramètre", 0, 10)

    def display_results(self):
        st.write(f"Résultat du modèle {self.name} avec paramètre {self.param}")

    def execute(self, data):
        pass