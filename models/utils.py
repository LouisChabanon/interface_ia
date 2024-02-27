import streamlit as st


class ModelType():
    REGRESSION = 1
    CLASSIFICATION = 2


class Model():
    def __init__(self, name: str, type: ModelType, parameters: list):
        self.name = str(name)
        self.type = type
        self.parameters = parameters

    def __str__(self):
        return f"Model: {self.name}, Type: {self.type}, Parameters: {self.parameters}"

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_parameters(self):
        return self.parameters

    def execute(self, data):
        print(f"Running model {self.name} with parameters {self.parameters}")

    def display_parameters(self):
        st.write("Aucun paramètre à afficher pour ce modèle")

    def display_results(self):
        st.write("Aucun résultat à afficher pour ce modèle")
