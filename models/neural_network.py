from models.utils import Model, ModelType, split_data
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.metrics import confusion_matrix


class NN(Model):
    def __init__(self):
        super().__init__("Réseaux de neuronnes", ModelType.CLASSIFICATION, None)
        self.param = {}
        self.data = None
        self.name = "Neural Network"


    def run(self):
        #ligne temporaire  
        self.data = pd.read_csv("./data/class_mushrooms.csv", sep=None, engine="python")
        
        X = self.data.drop(columns=['class'])
        y = self.data['class']

        # Convertir les entrées en encodage one-hot
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=1-self.param["ratio"])

        clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1)
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_test)
        return y_test, y_predicted


    def display_parameters(self):
        self.param["ratio"] = st.slider("Ratio d'entrainement", 0.1, 1.0, 0.8)
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            self.param["solver"] = st.selectbox('Choix du solveur', ('lbfgs', 'sgd', 'adam'))
            self.param["hidden_layers_sizes"] = tuple(map(int, eval(st.text_input("Nombre de neuronnes dans les couches cachées", value="(10, 5)"))))
        with col2:
            self.param["alpha"] = st.number_input("Terme de régularisation L2", value=1e-5, step=1e-6, format="%.6f")
            self.param["random_state"] = st.number_input("Random state", value=1.0, step=0.1, format="%.1f")


    def display_results(self, result):
        print(self.param)
        conf_matrix = confusion_matrix(result[0], result[1])
        # st.write("Résultats du réseau de neuronnes")
        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
        # ax.set_xlabel('Predicted')
        # ax.set_ylabel('True')
        # ax.set_title('Confusion Matrix')
        # plt.tight_layout()
        # st.pyplot(fig)
        st.write(conf_matrix)