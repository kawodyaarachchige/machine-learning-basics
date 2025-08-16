# Importing the required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data as a dictionary
# We are trying to fit a line through these x and y values
data = {
    'x': [1, 2, 3, 4, 5],  # Independent variable (input)
    'y': [2, 4, 6, 7, 10]   # Dependent variable (output)
}

# Creating a DataFrame (like an Excel table) using pandas
df = pd.DataFrame(data)
print(df)  # Show the dataset

# Extracting the feature column 'x' from the DataFrame
# It must be a 2D array (so use double brackets)
X = df[['x']]
print(X)
print(type(X))  # X is a DataFrame (2D)

# Extracting the target column 'y' (this is what we want to predict)
y = df['y']
print(y)
print(type(y))  # y is a Series (1D)

# Creating a Linear Regression model instance
model = LinearRegression()

# Training the model with the given data (X and y)
model.fit(X, y)

print("\nModel learned:")
print("Coefficient (slope):", model.coef_[0])     # Price increase per unit square footage
print("Intercept:", model.intercept_)             # Base price when square footage is 0

# 🏠 Prediction example: Estimate price for 600 sqft (i.e., x = 6)
predicted_price = model.predict([[7]])
print("\nPredicted price : ", predicted_price[0])