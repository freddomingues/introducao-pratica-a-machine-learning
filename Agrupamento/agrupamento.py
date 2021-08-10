import pandas as pd
base = pd.read_csv('Absenteeism_at_work.csv', sep=',')

from collections import Counter
contagem_razoes = Counter(base['Reason for absence'])
contagem_razoes

razoes = [13, 11, 19]
dados = base.loc[base['Reason for absence'].isin(razoes)]
colunas = dados.iloc[:, [6, 9]].values

from sklearn.preprocessing import StandardScaler
colunas_escalonadas = StandardScaler().fit_transform(colunas)

''' Elbow Method ou Método do Cotovelo
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(colunas_escalonadas)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.title('Elbow method para escolher o número de clusters')
plt.show()
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=4, random_state=0)
previsoes = kmeans.fit_predict(colunas_escalonadas)

for i in range(0, 121):
    if previsoes[i] == 0:
        plt.scatter(colunas[i, 0], colunas[i, 1], c='red')
    elif previsoes[i] == 1:
        plt.scatter(colunas[i, 0], colunas[i, 1], c='green')
    elif previsoes[i] == 2:
        plt.scatter(colunas[i, 0], colunas[i, 1], c='orange')
    else:
        plt.scatter(colunas[i, 0], colunas[i, 1], c='blue')

plt.xlabel('Distância entre casa e trabalho')
plt.ylabel('Carga de trabalho')
