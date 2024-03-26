from utils import Model, ModelType
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.metrics import confusion_matrix


class rdn(Model):
    def __init__(self):
        super().__init__("Réseaux de neuronnes", ModelType.CLASSIFICATION, None)

    def run(self, separated_data: pd.DataFrame):
        X = separated_data.drop(columns=['class'])
        y = separated_data['class']

        # Convertir les lettres en encodage one-hot
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1)
        clf.fit(X_train, y_train)

        y_test2 = clf.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_test2)

        # Afficher la matrice de confusion sous forme de heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        

def read_data():
    data = "./data/class_mushrooms.csv"
    data = pd.read_csv(data, sep=None, engine="python")
    return data

reseaux = rdn()
data = read_data()
rdn.run(rdn, data)