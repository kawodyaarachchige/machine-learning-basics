import pandas as pd                        # For data manipulation and analysis
import matplotlib.pyplot as plt            # For basic plotting
import seaborn as sns                      # For attractive statistical plots
import numpy as np                         # For numerical operations


# 📊 For building linear regression model and splitting data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 🎨 Set a clean visual style for seaborn plots
sns.set_style('whitegrid')

# 📁 Read the CSV file into a DataFrame
# This dataset contains advertising spend in different media and resulting sales
df = pd.read_csv('./assets/advertising.csv')

# 🖨️ Print the first 5 rows of the dataset to understand its structure
print(df.head())

# 📊 Create pairwise scatter plots between all numerical columns
# This helps visualize relationships (e.g., how TV ad spend correlates with Sales)
sns.pairplot(df)

# 🖼️ Display the plots
'''
    plt.show()
'''

x= df[['TV']]
print(x)

y= df[['Sales']]
print(y)

# 🔀 Split the dataset into training and testing sets
# 80% for training, 20% for testing; random_state=42 ensures reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ℹ️ Show the shapes of training and testing sets
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# 🧠 Create an instance of the Linear Regression model
model = LinearRegression()

# 🏗️ Train the model using the training data (TV vs. Sales)
model.fit(x_train, y_train)

# 📌 Display the learned slope and intercept of the line
# Equation form: Sales = (slope * TV) + intercept
print("\n📌 Model learned:")
print("Coefficient (slope):", model.coef_[0])     # How much Sales increases per unit increase in TV spend
print("Intercept:", model.intercept_)             # Estimated Sales when TV spend is 0          

# Plot the regression line on the test data
'''
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test, y_test, color='blue', label='Actual Sales')
    plt.plot(x_test, model.predict(x_test), color='red', linewidth=2, label='Predicted Sales (Regression Line)')
    plt.title('TV Ad Spending vs. Sales (Test Set)')
    plt.xlabel('TV Ad Spending ($ thousands)')
    plt.ylabel('Sales (thousands of units)')
    plt.legend()
    plt.show()
'''

# 🎯 Make predictions on the test data
y_pred = model.predict(x_test)

mse_value = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse_value:.4f}")
print(type(mse_value)) # Output: <class 'numpy.float64'>
rmse_value = np.sqrt(mse_value)
print(f"Root Mean Squared Error (RMSE): {rmse_value:.4f}")