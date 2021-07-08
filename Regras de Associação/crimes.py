# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:20:49 2021

@author: fred_
"""
import pandas as pd

base = pd.read_csv('https://raw.githubusercontent.com/fclesio/learning-space/master/Datasets/01%20-%20Association%20Rules/Crimes.csv')
f = base.describe()
transacoes = []

for i in range(0,780):
    transacoes.append([str(base.values[i,j]) for j in range(0,16)])
        
from apyori import apriori

regras = apriori(transacoes, min_support = 0.1, min_confidence = 0.7, min_lift = 2, min_lenght = 2)

resultados = list(regras)
resultados

resultados2 = [list(x) for x in resultados]
resultados2

resultadoFormatado = []
for j in range(0,6):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])
resultadoFormatado