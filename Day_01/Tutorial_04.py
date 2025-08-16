import pandas as pd                       
import matplotlib.pyplot as plt            
import seaborn as sns                     
import numpy as np                        

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

df = pd.read_csv('./assets/advertising.csv')

x= df[['TV','Radio','Newspaper']]

y= df[['Sales']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


print("Linear Regression")

model = LinearRegression()

model.fit(x_train, y_train)

print("Coefficient (slope):", model.coef_[0])    
print("Intercept:", model.intercept_)              

y_pred = model.predict(x_test)
mse_value = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse_value:.4f}")
rmse_value = np.sqrt(mse_value)
print(f"Root Mean Squared Error (RMSE): {rmse_value:.4f}")




