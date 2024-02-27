import pandas as pd
import matplotlib.pyplot as plt
from models.utils import separate_data


def test_regression_lineaire_run():
    from models.regression_lineaire import RegressionLineaire
    model = RegressionLineaire()
    data = pd.read_csv("./exemple.csv", sep=";")
    data = separate_data(data, 0.8)
    param = {"x_index": "Heure", "y_index": "NbVehicules"}
    prediction = model.run(data, param)

    print(prediction)
    plt.scatter(data["predict_data"]["Heure"],
                data["predict_data"]["NbVehicules"], color="black")
    plt.plot(data["predict_data"]["Heure"],
             prediction[:], color="blue")

    plt.show()


test_regression_lineaire_run()
