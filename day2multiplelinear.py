import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('file/advertising.csv')

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape",(x_train.shape, y_train.shape))

model=LinearRegression()
model.fit(x_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

y_pred = model.predict(x_test)

mse_value = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse_value:.4f}")

rmse_value = np.sqrt(mse_value)
print(f"Root Mean Squared Error: {rmse_value:.4f}")

