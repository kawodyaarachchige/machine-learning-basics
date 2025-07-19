import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns

# data ={
#     'x': [1, 2, 3, 4, 5],
#     'y': [2, 4, 6, 7, 10]
# }
# df = pd.DataFrame(data)
# print(df)


# x= df[['x']]
# print(x)
# print(type(x))

# y=df['y']
# print(y)
# print(type(y))

# model = LinearRegression()
# model.fit(x, y)

# print("Slope:", model.coef_[0])
# print("Intercept:", model.intercept_)


# #when x is 7000
# y_pred = model.predict([[7]])
# print("Predicted y:", y_pred)


# selles prediction by using csv file
df = pd.read_csv('advertising.csv')
x= df[['TV']]
y=df['Sales']
model = LinearRegression()
model.fit(x, y)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
y_pred = model.predict([[7]])
print("Predicted y:", y_pred)





