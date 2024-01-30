#Importation des modules nécessaires
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#from sklearn.preprocessing import scale

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


# Données d'entrainement (98%) et données de test (2%)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.05, shuffle=False)

print('train :', X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Initialize kNN
knn = KNeighborsRegressor(n_neighbors=10)

# Fit and score
knn.fit(X_train, y_train)

knn.score(X_test, y_test)
print("SCORE :", knn.score)

# Prédiction et contrôle de précision

y_pred = knn.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
print("MSE: ", MSE)



x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="JEU DE TEST")
plt.plot(x_ax, y_pred, linewidth=1, label="PREDICTION")
plt.title("KNN_y-test and y-predicted data")
plt.xlabel('JOURS')
plt.ylabel('FLUX')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)


plt.show()




def mae(x,y):
    M=0
    for i in range(len(x)):
        M+=np.abs(x[i]-y[i])
    return M/len(x)
    
def rmse(x,y) :
    R=0
    for i in range(len(x)):
        R+=(x[i]-y[i])**2
    return (R/len(x))**1/2
    
def mean(x):
    m=0
    for i in range(len(x)):
        m+=i
    return m/len(x)

def coeff_determination(x,y):
    S=0
    s=0
    m= mean(x)
    for i in range(len(x)):
        S+=(x[i]-y[i])**2
        s+=(x[i]-m)**2
    return 1-S/s


print("MAE(y_test, y_pred): ", mae(y_test.tolist(), y_pred.tolist()))
print("RMSE(y_test, y_pred): ", rmse(y_test.tolist(), y_pred.tolist()))
print("Mean données(y_test): ", mean(y_test.tolist()))
print("COEFF_DETERMINATION(y_test, y_pred): ", coeff_determination(y_test.tolist(), y_pred.tolist()))