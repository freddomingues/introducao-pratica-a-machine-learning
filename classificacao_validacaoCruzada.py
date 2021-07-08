# -*- coding: utf-8 -*-
"""
@author: fred_
"""

import pandas as pd
base = pd.read_csv("glass.csv")

x = base.iloc[:, 0:9].values
y = base.iloc[:, [9]].values

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state = 0)

from sklearn.naive_bayes import GaussianNB
resultadosNB = []
for indice_treino, indice_teste in kfold.split(x,np.ravel(y)):
    classificadorNB = GaussianNB()
    classificadorNB = classificadorNB.fit(x[indice_treino], y[indice_treino])
    previsoesNB = classificadorNB.predict(x[indice_teste])
    precisaoNB = accuracy_score(y[indice_teste], previsoesNB)
    resultadosNB.append(precisaoNB)
resultadosNB = np.asarray(resultadosNB)
resultadosNB.mean()

from sklearn.linear_model import LogisticRegression
resultadosRL = []
for indice_treino, indice_teste in kfold.split(x, np.ravel(y)):
    classificadorRL = LogisticRegression(max_iter = 2000)
    classificadorRL = classificadorRL.fit(x[indice_treino], y[indice_treino])
    previsoesRL = classificadorRL.predict(x[indice_teste])
    precisaoRL = accuracy_score(y[indice_teste], previsoesRL)
    resultadosRL.append(precisaoNB)
resultadosRL = np.asarray(resultadosRL)
resultadosRL.mean()

from sklearn.svm import SVC
resultadosSVM = []
for indice_treino, indice_teste in kfold.split(x, np.ravel(y)):
    classificadorSVM = SVC(kernel='linear')
    classificadorSVM= classificadorSVM.fit(x[indice_treino], y[indice_treino])
    previsoesSVM = classificadorSVM.predict(x[indice_teste])
    precisaoSVM = accuracy_score(y[indice_teste], previsoesSVM)
    resultadosSVM.append(precisaoSVM)
resultadosSVM = np.asarray(resultadosSVM)
resultadosSVM.mean()

















