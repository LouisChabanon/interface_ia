import streamlit as st
import pandas as pd


class ModelType():
    REGRESSION = 1
    CLASSIFICATION = 2


class Model():
    def __init__(self, name: str, type: ModelType, parameters: list):
        self.name = str(name)
        self.type = type
        self.parameters = parameters
        self.data = None

    def setparameters(self, parameters):
        self.parameters = parameters
    
    def setdata(self, data):
        self.data = data

    def __str__(self):
        return f"Model: {self.name}, Type: {self.type}, Parameters: {self.parameters}"

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_parameters(self):
        return self.parameters


    def run(self, data):
        pass


    def display_parameters(self):
        st.write("Aucun paramètre à afficher pour ce modèle")

    def display_results(self):
        st.write("Aucun résultat à afficher pour ce modèle")


def split_data(data: pd.DataFrame, ratio: float):
    training_data = data.sample(frac=ratio, random_state=0)
    predict_data = data.drop(training_data.index)
    return {"training_data": training_data, "predict_data": predict_data}

def display_data(data: pd.DataFrame):
    return st.dataframe(data)