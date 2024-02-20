import streamlit as st
import pandas as pd
import base64


# test
exemple = "./exemple.csv"

def display_pdf(file_path):
    """Affiche le PDF dans l'application Streamlit en utilisant une iframe HTML."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def fiches_page():
    st.title("Fiches explicatives")
    st.header("Méthodes de Modélisation")

    tab1, tab2 = st.tabs(["Classification", "Régression"])

    methodes_classification = {"K mean": "fiche_k_mean.pdf","KNN": "fiche_KNN.pdf"}

    methodes_regression = {"Regression Linéaire": "fiche_regression_linéaire.pdf","Regression Polynomiale": "fiche_regression_polynomiale.pdf"}

    with tab1:
        st.subheader("Classification")
        methode_choisie = st.selectbox("Choisir une méthode de classification", list(methodes_classification.keys()))
        display_pdf(methodes_classification[methode_choisie])

    with tab2:
        st.subheader("Régression")
        methode_choisie = st.selectbox("Choisir une méthode de régression", list(methodes_regression.keys()))
        display_pdf(methodes_regression[methode_choisie])


def main() -> None:
    st.title("Interface IA")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisir une page", ["Page Principale", "Fiches"])

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
        st.download_button(label="Télécharger", data=exemple,file_name="exemple.csv")
    with col2:
        exemple_check = st.checkbox("Utiliser l'exemple")
    if exemple_check:
        data = exemple
    if data:
        data = pd.read_csv(data)

    st.header("Paramétrer votre modèle")
    st.subheader("Choix du modèle")
    categorie = st.selectbox("Choisir le type de modele", ["Regression", "Classification"])
    
    if categorie == "Regression":
        model = st.selectbox("Choisir le modèle", ["Regression linéaire", "Regression polynomiale"])
    elif categorie == "Classification":
        model = st.selectbox("Choisir le modèle", ["K mean", "KNN"])

    st.subheader("Choix des paramètres")
    if model == "Regression linéaire":
        param = st.slider("Choisir le paramètre", 0, 10)

    st.header("Visualiser les résultats")


if __name__ == "__main__":
    main()
