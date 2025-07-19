import pandas as pd
#from data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#for building the model and evaluating it
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#set a style for seaborn
sns.set(style="whitegrid")

data = pd.read_csv("advertising.csv")


#sns.pairplot(data)
#show the plot
#plt.show()

x= data[['TV']]
print(x)

y=data['Sales']
print(y)

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

sns.scatterplot(x_test, y_test)
sns.lineplot(x_test, y_pred)
plt.show()


