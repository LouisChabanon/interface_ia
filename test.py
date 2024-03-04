import pandas as pd
import matplotlib.pyplot as plt
from models.utils import separate_data


def test_regression_lineaire_run():
    from models.regression_lineaire import RegressionLineaire
    model = RegressionLineaire()
    data = pd.read_csv("./exemple.csv", sep=";")
    param = ["Heure", "NbVehicules", 0.8]
    prediction = model.run(data, param)

    print(prediction)


test_regression_lineaire_run()
