import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for building the model and evaluating it
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

# set a style for our plots
sns.set_style("whitegrid")

df=pd.read_csv('file/advertising.csv')
print(df.head())


# create graph
# sns.pairplot(df)
# plt.show()

X = df[['TV']]
y = df['Sales']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape",(x_train.shape, y_train.shape)) 

# create scaler object
scaler = StandardScaler()

# fit the scaler on the training data and transform both training and test data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

sgd_model = SGDRegressor(max_iter=5000, random_state=42)
sgd_model.fit(x_train_scaled, y_train)

intercept = sgd_model.intercept_
coefficient = sgd_model.coef_
print("Intercept:", intercept)
print("Coefficient:", coefficient)

# calculate rmse 
y_pred = sgd_model.predict(x_test_scaled)
mse_value = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse_value:.4f}")
rmse_value = np.sqrt(mse_value)
print(f"Root Mean Squared Error: {rmse_value:.4f}")
