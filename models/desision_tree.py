from models.utils import Model, ModelType
import streamlit as st
import pandas as pd


# Faire plusieurs exemples

class DecisionTree(Model):
    def __init__(self, name: str, type: ModelType, parameters: list):
        super().__init__(name, type, parameters)
        self.param = None
