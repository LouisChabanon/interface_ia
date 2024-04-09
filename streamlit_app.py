import streamlit as st
import pandas as pd
from models.regression_lineaire import RegressionLineaire
from models.desision_tree import DecisionTree
from models.neural_network import NN
from models.KNN import KNN
from models.KMEAN import KMEAN
from models.Randomforest import RandomForest
import base64
from models.utils import display_data, split_data

exemple = "./exemple.csv"


def display_pdf(file_path):
    """Affiche le PDF dans l'application Streamlit en utilisant une iframe HTML."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def calculer_statistiques(df, colonnes_a_analyser):
    if df is not None and colonnes_a_analyser:
        try:
            df_filtre = df[colonnes_a_analyser]
            description = df_filtre.describe()
            variance = df_filtre.var()
            skewness = df_filtre.skew()
            kurtosis = df_filtre.kurtosis()
            iqr = description.loc['75%'] - description.loc['25%']
            somme = df_filtre.sum()

            with st.expander("Tendance Centrale"):
                st.write("Mesures de la tendance centrale des données.")
                for col in colonnes_a_analyser:
                    st.metric(label=f"Moyenne {col}", value=f"{description.loc['mean'][col]:.2f}",
                              help="""La moyenne des valeurs. Indique la tendance centrale des données. 
                                      Peut être influencée par des valeurs extrêmes.""")
                    st.metric(label=f"Médiane {col}", value=f"{description.loc['50%'][col]:.2f}",
                              help="La valeur centrale des données. Moins sensible aux valeurs extrêmes que la moyenne.")

            with st.expander("Dispersion"):
                st.write("Mesures de la dispersion des données.")
                for col in colonnes_a_analyser:
                    st.metric(label=f"Écart type {col}", value=f"{description.loc['std'][col]:.2f}",
                              help="""Mesure de la dispersion des données autour de la moyenne. 
                                      Une valeur élevée indique une grande variabilité des données.""")
                    st.metric(label=f"Variance {col}", value=f"{variance[col]:.2f}",
                              help="Le carré de l'écart type. Mesure également la dispersion des données.")
                    st.metric(label=f"IQR {col}", value=f"{iqr[col]:.2f}",
                              help="Intervalle interquartile. Différence entre le 75e et le 25e percentile. Indique la dispersion du milieu 50% des données.")

            with st.expander("Forme de la Distribution"):
                st.write(
                    "Mesures décrivant la forme de la distribution des données.")
                for col in colonnes_a_analyser:
                    st.metric(label=f"Asymétrie {col}", value=f"{skewness[col]:.2f}",
                              help="""Mesure de l'asymétrie de la distribution des données. 
                                      Une valeur > 0 indique une queue plus longue à droite, < 0 à gauche.""")
                    st.metric(label=f"Curtose {col}", value=f"{kurtosis[col]:.2f}",
                              help="""Mesure du degré de pointe de la distribution des données. 
                                      Une valeur élevée indique une distribution plus pointue.""")

        except Exception as e:
            st.error(f"Erreur lors du calcul des statistiques : {e}")


def fiches_page():
    st.title("Fiches explicatives")
    st.header("Méthodes de Modélisation")

    tab1, tab2 = st.tabs(["Classification", "Régression"])

    methodes_classification = {
        "K mean": "fiche_k_mean.pdf", "KNN": "fiche_KNN.pdf"}

    methodes_regression = {
        "Regression Linéaire": "fiche_regression_linéaire.pdf",
        "Regression Polynomiale": "fiche_regression_polynomiale.pdf",
    }

    with tab1:
        st.subheader("Classification")
        methode_choisie = st.selectbox(
            "Choisir une méthode de classification",
            list(methodes_classification.keys()),
        )
        display_pdf("./fiches/" + methodes_classification[methode_choisie])

    with tab2:
        st.subheader("Régression")
        methode_choisie = st.selectbox(
            "Choisir une méthode de régression", list(
                methodes_regression.keys())
        )
        display_pdf(methodes_regression[methode_choisie])


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

    # Upload Données
    st.header("Uploader vos données")
    data = st.file_uploader("Uploader un dataset", type=["csv"])
    st.write("ou utiliser l'exemple ci-dessous")
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.download_button(label="Télécharger", data=exemple,
                           file_name="exemple.csv")
    with col2:
        exemple_check = st.checkbox("Utiliser l'exemple", value=True)
    if exemple_check:
        data = exemple
    if data:
        # Autodetect separator
        data = pd.read_csv(data, sep=None, engine="python")
        display_data(data)

        colonnes_numeriques = data.select_dtypes(
            include=['float64', 'int64']).columns.tolist()
        colonnes_a_analyser = st.multiselect("Sélectionnez les colonnes à analyser", options=colonnes_numeriques, default=colonnes_numeriques,
                                             help="Sélectionnez les colonnes numériques pour lesquelles vous souhaitez calculer des statistiques.")

        if st.button('Calculer les statistiques descriptives'):
            calculer_statistiques(data, colonnes_a_analyser)
        st.header("Paramétrer votre modèle")
        st.subheader("Choix du modèle")
        categorie = st.selectbox(
            "Choisir le type de modele", ["Regression", "Classification"]
        )
        algorithm = None
        if categorie == "Regression":
            model = st.selectbox(
                "Choisir le modèle", [
                    "Regression linéaire", "Regression polynomiale"]
            )
        elif categorie == "Classification":
            model = st.selectbox(
                "Choisir le modèle",
                ["K mean", "KNN", "Random Forest",
                    "Neural Network", "Decision Tree"],
            )

        if model == "Regression linéaire":
            algo = RegressionLineaire()
        elif model == "KNN":
            algo = KNN()
        elif model == "Decision Tree":
            algo = DecisionTree()
        elif model == "K mean":
            algo = KMEAN()
        elif model == "Neural Network":
            algo = NN()
        elif model == "Random Forest":
            algo = RandomForest()

        algo.setdata(data)
        st.subheader("Choix des paramètres")
        algo.display_parameters()
        st.header("Exectuer le modèle")
        execution = st.button("Exécuter")
        if execution:
            with st.spinner("Exécution du modèle..."):
                result = algo.run()
            st.header("Visualiser les résultats")
            algo.display_results(result)


if __name__ == "__main__":
    main()
