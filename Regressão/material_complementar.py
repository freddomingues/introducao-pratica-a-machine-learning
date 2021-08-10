import pandas as pd
base = pd.read_csv('insurance.csv')

base.describe()

x = base.iloc[:, 0:6].values
y = base.iloc[:, [6]].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x[:, 1] = encoder.fit_transform(x[:, 1])
x[:, 4] = encoder.fit_transform(x[:, 4])
x[:, 5] = encoder.fit_transform(x[:, 5])

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=0)

# MAE regressão linear = 3998
# MAE polinomial degree 2 = 2864
# MAE arvore = 3260

''' Regressão polinomial
from sklearn.preprocessing import PolynomialFeatures
pol_transformacao = PolynomialFeatures(degree=3)
pol_x_treino = pol_transformacao.fit_transform(x_treino)
pol_x_teste = pol_transformacao.fit_transform(x_teste)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(pol_x_treino, y_treino)
previsoes = regressor.predict(pol_x_teste)

from sklearn.metrics import mean_absolute_error
MAE_pol = mean_absolute_error(previsoes, y_teste)
'''

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

regressor.fit(x_treino, y_treino)
score_treino = regressor.score(x_treino, y_treino)
score_teste = regressor.score(x_teste, y_teste)
previsao = regressor.predict(x_teste)

from sklearn.metrics import mean_absolute_error
MAE_arvore = mean_absolute_error(previsao, y_teste)
print(MAE_arvore)
