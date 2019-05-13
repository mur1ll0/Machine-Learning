import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split

#Carregar DB iris
iris = ds.load_iris()

# 70% training and 30% test
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.3, random_state=4)


#Imprimir dados treino (x,y)=(2,3)
plt.scatter(Xtr[ytr==0,2],Xtr[ytr==0,3], color='r',label=iris.target_names[0])
plt.scatter(Xtr[ytr==1,2],Xtr[ytr==1,3], color='g',label=iris.target_names[1])
plt.scatter(Xtr[ytr==2,2],Xtr[ytr==2,3], color='b',label=iris.target_names[2])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.legend()
plt.show()

#Montando aprendizado de maquina
neighbors = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

#Treinar
neighbors.fit(Xtr,ytr)

#Testar
y_hat = neighbors.predict(Xte)


#Imprimir dados teste (x,y)=(2,3)
plt.scatter(Xtr[ytr==0,2],Xtr[ytr==0,3], color='r',label=iris.target_names[0])
plt.scatter(Xtr[ytr==1,2],Xtr[ytr==1,3], color='g',label=iris.target_names[1])
plt.scatter(Xtr[ytr==2,2],Xtr[ytr==2,3], color='b',label=iris.target_names[2])

plt.scatter(Xte[y_hat==0,2],Xte[y_hat==0,3], color='r', marker='^', label=iris.target_names[0])
plt.scatter(Xte[y_hat==1,2],Xte[y_hat==1,3], color='g', marker='^', label=iris.target_names[1])
plt.scatter(Xte[y_hat==2,2],Xte[y_hat==2,3], color='b', marker='^', label=iris.target_names[2])

plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.legend()
plt.show()
