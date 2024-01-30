#Importation des modules nécessaires

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
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

# =============================================================================
# affichage courbes flux
# =============================================================================

YY = pd.DataFrame(Y)
YY.plot(grid=True, figsize=(20, 10))
plt.title('Flux')
plt.show()



# Données d'entraînement (99%) et données de test (1%)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5, shuffle=False)

print('train :', X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#Déclaration de l'arbre de décision OPTIMISE
DTR = DecisionTreeRegressor(ccp_alpha=0.0, criterion='squared_error', max_depth=7,
                      max_features=None, 
                      max_leaf_nodes=None,
                      min_impurity_decrease=0.0,
                      min_samples_leaf=2, min_samples_split=3,
                      min_weight_fraction_leaf=0.0, 
                      random_state=None, 
                      splitter='best')


#Entrainement de l'abre de décision 
DTR.fit(X_train, y_train)

#précision du modèle

score = DTR.score(X_train, y_train)
print("R-squared:", score) 

#Affichage de l'abre de décision obtenu après entrainement
# plot_tree(DTR, feature_names= ['Num_jour','Heure','WE','HBureau','Vacances','HScolaire'], class_names=["NbVehicules"],filled=True, fontsize=(7))
# plt.savefig('IMAGES\\arbre')
# plt.show()

# Visualisation du modèle d'arbre de décision

export_graphviz(DTR, out_file='tree.dot', feature_names= ['Num_jour','Heure','WE','HBureau','Vacances','HScolaire'], 
                class_names=["NbVehicules"], filled = True)



# Prédiction et contrôle de précision

y_pred = DTR.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
print("MSE: ", MSE)
print("RMSE: ", MSE**(1/2.0)) 


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=2, label="JEU DE TEST")
plt.plot(x_ax, y_pred, linewidth=1.1, label="PREDICTION")
plt.title("decison tree_y-test and y-predicted data NON OPTIMIZED")
plt.xlabel('JOURS')
plt.ylabel('FLUX')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)

plt.show() 

# =============================================================================
# optimisation
# =============================================================================
DTR_opt = DecisionTreeRegressor()
std_slc = StandardScaler()
pca = decomposition.PCA()
pipe = Pipeline(steps=[("std_slc", std_slc),
                       ("pca", pca),
                       ("DTR", DTR_opt)])

n_components = list(range(1,X.shape[1]+1,1))
    
criterion = ["friedman_mse", 'squared_error']
max_depth = [2,3,4,5,6,7,8,9,10]

parameters = dict(pca__n_components=n_components,
                  DTR__criterion=criterion,
                  DTR__max_depth=max_depth)

clf = GridSearchCV(pipe, parameters)
clf.fit(X, Y)


print("Best Number Of Components:", clf.best_estimator_.get_params()["pca__n_components"])
print(); print(clf.best_estimator_.get_params()["DTR"])

CV_Result = cross_val_score(clf, X, Y, cv=5, n_jobs=-1, scoring="r2")
print(); print('CV_Result :', CV_Result)
print(); print('CV_Result.mean :', CV_Result.mean())
print(); print('CV_Result.std :', CV_Result.std())

#Déclaration de l'arbre de décision OPTIMISE
DTR_opt = DecisionTreeRegressor(ccp_alpha=0.0, criterion='friedman_mse', max_depth=4,
                      max_features=None, 
                      max_leaf_nodes=None,
                      min_impurity_decrease=0.0,
                      min_samples_leaf=2, min_samples_split=3,
                      min_weight_fraction_leaf=0.0, 
                      random_state=None, 
                      splitter='best')


#Entrainement de l'abre de décision 
DTR_opt.fit(X_train, y_train)

#précision du modèle

score = DTR_opt.score(X_train, y_train)
print("R-squared_opt:", score) 

#Affichage de l'abre de décision obtenu après entrainement
# plot_tree(DTR, feature_names= ['Num_jour','Heure','WE','HBureau','Vacances','HScolaire'], class_names=["NbVehicules"],filled=True, fontsize=(7))
# plt.savefig('IMAGES\\arbre')
# plt.show()
# Visualisation du modèle d'arbre de décision

export_graphviz(DTR_opt, out_file='tree_opt.dot', feature_names= ['Num_jour','Heure','WE','HBureau','Vacances','HScolaire'], 
                class_names=["NbVehicules"], filled = True)


# Prédiction et contrôle de précision

y_pred_opt = DTR_opt.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
print("MSE_opt: ", MSE)
print("RMSE_opt: ", MSE**(1/2.0)) 


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=2, label="JEU DE TEST")
plt.plot(x_ax, y_pred, linewidth=1.1, label="PREDICTION")
plt.plot(x_ax, y_pred_opt, linewidth=1.1, label="PREDICTION_opt")
plt.title("Decision tree_y-test and y-predicted data OPTIMIZED")
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