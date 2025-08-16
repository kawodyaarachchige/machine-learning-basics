import ssl
# For data loading and manipulation
from sklearn.datasets import fetch_openml
import numpy as np

# For building and training the model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# For evaluation and visualization
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target
print("data set load successfully")

# print(X)
# print(type(X))
# print("shape of x:" , X.shape)

# print("--------------------------")

# print(y)
# print(type(y))
# print("shape of y:" , y.shape)


# lets visualize a single image
some_digit_index = 0
some_digit_image = X[some_digit_index].reshape(28, 28)
true_label = y[some_digit_index]

plt.imshow(some_digit_image, cmap='binary')
plt.title(f"True Label: {true_label}")
plt.axis('off')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# scale down pixels to [0, 1] range
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

print("data has been split and scaled down")
print("training set size :", len(X_train_scaled))
print("testing set size :", len(X_test_scaled))

# Train a softmax regression model
softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000 ,random_state=42)
softmax_reg.fit(X_train_scaled, y_train)
print("model has been trained")

#make predictions on the test set
y_pred = softmax_reg.predict(X_test_scaled)

#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.4f}") 

# Display the confusion matrix to see where the model made mistakes
fig,ax = plt.subplots(figsize=(10, 10))
print("ax",ax)

ConfusionMatrixDisplay.from_estimator(softmax_reg, X_test_scaled, y_test, ax=ax,cmap='Blues', normalize='true')
print("Confusion matrix displayed")
plt.show()