import streamlit as st
import pandas as pd
# test
exemple = "./exemple.csv"


def fiches_page():
    st.title("Fiches explicatives")
    st.header("Méthodes de Modélisation")
    methodes = {
        "Classification": {
            "K mean": "chemin_vers_image_k_mean.jpg",
            "KNN": "chemin_vers_image_KNN.jpg"},
        "Régression": {
            "Regression Linéaire": "chemin_vers_image_regression_linéaire.jpg",
            "Regression Polynomiale": "chemin_vers_image_regression_polynomiale.jpg"}}

    # Sélection du modèle
    modele = st.selectbox("Choisir un modèle", list(methodes.keys()))

    # Sélection de la méthode en fonction de la catégorie choisie
    methode_choisie = st.selectbox(
        "Choisir une méthode", list(methodes[modele].keys()))

    # Affichage de l'image correspondante à la méthode choisie
    st.image(methodes[modele][methode_choisie], caption=methode_choisie)


def main() -> None:
    st.title("Interface IA")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page", ["Page Principale", "Fiches"])

    if page == "Page Principale":
        main_page()
    elif page == "Fiches":
        fiches_page()


def main_page():
    st.header("Uploader vos données")
    data = st.file_uploader("Uploader un dataset", type=["csv"])
    st.write("ou utiliser l'exemple ci-dessous")
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.download_button(label="Télécharger", data=exemple,
                           file_name="exemple.csv")
    with col2:
        exemple_check = st.checkbox("Utiliser l'exemple")
    if exemple_check:
        data = exemple
    if data:
        data = pd.read_csv(data)

    st.header("Paramétrer votre modèle")
    st.subheader("Choix du modèle")
    categorie = st.selectbox("Choisir le type de modele", [
                             "Regression", "Classification"])
    if categorie == "Regression":
        model = st.selectbox("Choisir le modèle", [
                             "Regression linéaire", "Regression polynomiale"])
    elif categorie == "Classification":
        model = st.selectbox("Choisir le modèle", [
                             "K mean", "KNN"])

    st.subheader("Choix des paramètres")
    if model == "Regression linéaire":
        param = st.slider("Choisir le paramètre", 0, 10)

    st.header("Visualiser les résultats")


if __name__ == "__main__":
    main()
