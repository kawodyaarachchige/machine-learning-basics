import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("./assets/womens_clothing_ecommerce_reviews.csv")


X = df ['Review Text'] 
y = df ['sentiment'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#define the pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('classifier', LogisticRegression(max_iter=1000))
])


print("Training the entire pipeline...")
Pipeline.fit(X_train, y_train)
print("✅ Pipeline training complete.")