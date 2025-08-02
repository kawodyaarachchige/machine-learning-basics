import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # for text processing (feature vectorization)
from sklearn.linear_model import LogisticRegression # for classification
from sklearn.metrics import accuracy_score # for model evaluation

# Load the dataset
df = pd.read_csv('file/womens_clothing_ecommerce_reviews.csv', encoding='latin-1')
print(df.head())
print(df.info())