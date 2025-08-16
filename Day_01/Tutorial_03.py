import pandas as pd                        # For data manipulation and analysis
import matplotlib.pyplot as plt            # For basic plotting
import seaborn as sns                      # For attractive statistical plots
import numpy as np                         # For numerical operations


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn import metrics


df = pd.read_csv('./assets/advertising.csv')

X = df[['TV']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

sgd_model = SGDRegressor(max_iter=5000, random_state=42)
sgd_model.fit(X_train_scaled,y_train)

intercept = sgd_model.intercept_
coefficients = sgd_model.coef_

print(f"Intercept: {intercept[0]:.4f}") # Intercept: 15.3101
print(f"Coefficients: {coefficients[0]:.4f}") # Coefficients: 4.6634

y_pred = sgd_model.predict(X_test_scaled)

mse_value = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse_value:.4f}")
rmse_value = np.sqrt(mse_value)
print(f"Root Mean Squared Error (RMSE): {rmse_value:.4f}")
