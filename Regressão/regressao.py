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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treino, y_treino)
score_treino = regressor.score(x_treino, y_treino)

previsoes = regressor.predict(x_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error
MAE = mean_absolute_error(y_teste, previsoes)

MSE = mean_squared_error(y_teste, previsoes)

score_teste = regressor.score(x_teste, y_teste)
