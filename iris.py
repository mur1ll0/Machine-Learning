import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Arrumar arquivos para dizer se é Iris-setosa ou nao (0 e 1)
data = np.genfromtxt('iris.data', delimiter=',', dtype=str, skip_header=0)
classes = np.char.replace(data, 'Iris-setosa', '0')
classes = np.char.replace(classes, 'Iris-versicolor', '1')
classes = np.char.replace(classes, 'Iris-virginica', '2').astype(float)

#Emabaralhar dados, pois estao ordenados
np.random.seed(4)
np.random.shuffle(classes)

#Separar rótulo dos dados
y = classes[:,-1:]
X = classes[:,:-1]

#Dividir conjuto de dados 75% treino e 25% teste
nsize = int(X.shape[0]*.75)

Xtr = X[:nsize,:]
ytr = y[:nsize,:]
Xte = X[nsize:,:]
yte = y[nsize:,:]


#Montando aprendizado de maquina
iris = linear_model.LogisticRegression()
#Treinar
ytr = np.ravel(ytr) #Planificar rotulos
iris.fit(Xtr,ytr)

#Testar
yte = np.ravel(yte) #Planificar rotulos
y_hat = iris.predict(Xtr)
print('   Iris-setosa (treino):', np.mean((y_hat==0)==(ytr==0)))
print('   Iris-versicolor (treino):', np.mean((y_hat==1)==(ytr==1)))
print('   Iris-virginica (treino):', np.mean((y_hat==2)==(ytr==2)))
y_hat = iris.predict(Xte)
print('\n   Iris-setosa (teste):', np.mean((y_hat==0)==(yte==0)))
print('   Iris-versicolor (teste):', np.mean((y_hat==1)==(yte==1)))
print('   Iris-virginica (teste):', np.mean((y_hat==2)==(yte==2)))

