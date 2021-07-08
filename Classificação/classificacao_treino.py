# -*- coding: utf-8 -*-
"""
@author: fred_
"""

import pandas as pd
base = pd.read_csv("glass.csv")

x = base.iloc[:,0:9].values
y = base.iloc[:,[9]].values

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30)

import numpy as np
from sklearn.naive_bayes import GaussianNB
classificadorNB = GaussianNB()
modelo = classificadorNB.fit(x_treino, np.ravel(y_treino))
previsoesNB = modelo.predict(x_teste)

from sklearn.linear_model import LogisticRegression
classificadorRL = LogisticRegression(max_iter=2000)
classificadorRL = classificadorRL.fit(x_treino,np.ravel(y_treino))
previsoesRL = classificadorRL.predict(x_teste)

from sklearn.svm import SVC
classificadorSVM = SVC(kernel='linear')
classificadorSVM = classificadorSVM.fit(x_treino, np.ravel(y_treino))
previsoesSVM = classificadorSVM.predict(x_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
matrizNB = confusion_matrix(y_teste, previsoesNB)
acuraciaNB = accuracy_score(y_teste, previsoesNB)

matrizRL = confusion_matrix(y_teste, previsoesRL)
acuraciaRL = accuracy_score(y_teste, previsoesRL)

matrizSVM = confusion_matrix(y_teste, previsoesSVM)
acuraciaSVM = accuracy_score(y_teste, previsoesSVM)


