from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



#importation du dataset
données = pd.read_csv('DATA_2022_Ecole_ssdate_court.csv', delimiter=';')

# Affichage de données
données.head()

# Structure de la dataframe
données.info()

X = données.values[:, 0:6]
Y = données.values[:,6]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=28)

model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", rmse)


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="JEU DE TEST")
plt.plot(x_ax, y_pred, linewidth=1, label="PREDICTION")
plt.title("KNN_y-test and y-predicted data")
plt.xlabel('JOURS')
plt.ylabel('FLUX')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)


plt.show()