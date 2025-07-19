# Import the pandas library to work with data (tables like Excel)
import pandas as pd

# Import Matplotlib and Seaborn libraries for data visualization (graphs and plots)
import matplotlib.pyplot as plt
import seaborn as sns

# Import tools to split data and create a linear regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set the visual style for Seaborn plots to look neat with white background and grid
sns.set(style="whitegrid")

# Load the CSV file 'advertising.csv' into a DataFrame called 'data'
data = pd.read_csv("advertising.csv")

# Create a scatterplot matrix (commented out here; can be used for visual analysis)
# sns.pairplot(data)
# plt.show()

# Select the 'TV' column from the dataset as input (feature) and store in variable 'x'
x = data[['TV']]  # Double brackets keep x as a DataFrame
print(x)  # Print the input values (TV ad spending)

# Select the 'Sales' column as the output (target value we want to predict)
y = data['Sales']
print(y)  # Print the target values (sales numbers)

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42  # random_state ensures reproducibility
)

# Print the shape (rows and columns) of each split dataset
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Create the Linear Regression model
model = LinearRegression()

# Train (fit) the model using training data
model.fit(x_train, y_train)

# Print the slope (coefficient) of the regression line
print("Slope:", model.coef_[0])

# Print the intercept (where the line crosses the y-axis)
print(f"Intercept: {model.intercept_:2f}")

# # Use the model to make predictions on the test data
# y_pred = model.predict(x_test)
# print(y_pred)  # Show predicted sales values

# # Calculate residuals (difference between actual and predicted values)
# residuals = y_test - y_pred
# print(residuals)  # Show the errors (how far off the predictions were)

# # Create a scatter plot to compare actual sales vs predicted sales
# sns.scatterplot(x=y_test, y=y_pred)

# # Label the axes and add a title
# plt.xlabel("Actual Sales")
# plt.ylabel("Predicted Sales")
# plt.title("Actual vs Predicted Sales")

# # Show the plot
# plt.show()


# Plot the regression line on the test data
plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='blue', label='Actual Sales')
plt.plot(x_test, model.predict(x_test), color='red', linewidth=2, label='Predicted Sales (Regression Line)')
plt.title('TV Ad Spending vs. Sales (Test Set)')
plt.xlabel('TV Ad Spending ($ thousands)')
plt.ylabel('Sales (thousands of units)')
plt.legend()
plt.show()
