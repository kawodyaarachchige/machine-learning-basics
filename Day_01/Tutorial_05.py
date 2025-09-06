
import pandas as pd # Importing the pandas library to work with data in tables (like Excel)
from sklearn.model_selection import train_test_split # Importing a function to split our data into training and testing parts
from sklearn.feature_extraction.text import CountVectorizer # Importing a tool to convert text into numbers (so the computer can understand it)
from sklearn.linear_model import LogisticRegression # Importing a simple machine learning model called Logistic Regression (used for classification)
from sklearn.metrics import accuracy_score # Importing a tool to check how accurate our model's predictions are


df = pd.read_csv("./assets/womens_clothing_ecommerce_reviews.csv")

# This line prints the first 5 rows of the table so we can get a quick look at what kind of data it has.
print(df.head())


# This line gives a summary of the entire dataset.
# It shows:
# - How many rows and columns the data has
# - The name and type of each column (like text, number, etc.)
# - How many missing (empty) values are in each column
df.info()

X = df ['Review Text'] # We're trying to predict whether a review is positive or negative
y = df ['sentiment'] # The actual labels (the values we're trying to predict)


# This line splits our data into 4 parts:
# - X_train: the reviews we'll use to train (teach) the model
# - X_test: the reviews we'll use to test how good the model is
# - y_train: the correct answers (sentiments) for the training reviews
# - y_test: the correct answers for the testing reviews

# test_size=0.2 means 20% of the data is saved for testing, and 80% is used for training.
# random_state=42 just makes sure we get the same split every time we run the code.
# stratify=y makes sure both positive and negative reviews are evenly split in training and testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print ("\n Converting text to numerical features using Bag-of-words approach...")

# We are creating something called a "CountVectorizer".
# It turns text (like full sentences) into numbers that a computer can understand.
# stop_words='english' means it will ignore common words like "the", "is", "and", etc.,
# because they don’t help much in understanding the meaning of a review.
vectorizer = CountVectorizer(stop_words='english')

# This line teaches the vectorizer what words appear in the training reviews
# and then transforms (converts) each review into a bag-of-words format (numbers based on word counts).
X_train_bow = vectorizer.fit_transform(X_train)

# This transforms the test reviews into the same bag-of-words format
# using the words the vectorizer learned from the training data.
X_test_bow = vectorizer.transform(X_test)

print ("\n Text successfully converted to feature vectors using Bag-of-words approach.")


# We're creating a machine learning model called Logistic Regression.
# It will help us decide whether a review is positive or negative.
# max_iter=2000 means it can take up to 2000 steps to find the best solution (just to be safe).
model = LogisticRegression(max_iter=2000)

# Now we train the model using our training data.
# We're teaching it what words are in the review and what the correct sentiment is (positive or negative).
model.fit(X_train_bow, y_train)
print("\n Model successfully trained.")

# Now we use the model to predict the sentiment of the test reviews (the 20% we saved earlier).
y_pred = model.predict(X_test_bow)

# This line checks how many predictions the model got right out of all the test reviews.
# It gives a number between 0 and 1. (For example: 0.87 means 87% correct)
accuracy = accuracy_score(y_test, y_pred)

# This line prints the accuracy in two ways:
# - Four decimal places (like 0.8743)
# - As a percentage (like 87.43%)
print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")


reviews = [
    "I love this dress, it fits perfectly!",
    "The material feels cheap and uncomfortable.",
    "This shirt is so soft and looks amazing."
    "Not happy with the size, it's way too small."
    "Absolutely beautiful and worth the price."
]

X_test_bow = vectorizer.transform(reviews)

y_pred = model.predict(X_test_bow)

for i in range(len(reviews)):
    print(f"Review: {reviews[i]}")
    print(f"Sentiment: {y_pred[i]}\n")

# This line saves the trained model to a file called "model.joblib"
joblib.dump(model, 'model.joblib')


from google.cloud import aiplatform

aiplatform.init(
    project="aqueous-thought-470603-m5",
    location="us-central1"
)

endpoint = aiplatform.Endpoint(
    endpoint_name='8598324965231558656'
)

print("✅ Endpoint is ready to predict..")


new_review = ["The material felt cheap and it was not what I expected."]

print(f"New review to classify :",{new_review[0]})

print("Step 1 : converting to a sparse matrix...")
sparse_matrix = vectorizer.transform(new_review)

print("Step 2 : converting to a dense Numpy array...")
numpy_array = sparse_matrix.toarray()

print(" Step 3 : converting to a list for API call...")
processed_review = numpy_array.tolist()

print("\n Sending prediction request to the Vertex AI endpoint...")

# Make the prediction
response = endpoint.predict(instances=processed_review)

# Print the prediction
print(response)
print(type(response))


'''
new_review = ["The material felt cheap and it was not what I expected.",
              "The dress fit perfectly and was very comfortable.",
              "The shirt was so soft and look amazing.",
              "The size was way too small.",
              "Absolutely beautiful and worth the price."
              ]

for review in new_review:
    print(f"New review to classify :",{review})

    print("Step 1 : converting to a sparse matrix...")
    sparse_matrix = vectorizer.transform([review])

    print("Step 2 : converting to a dense Numpy array...")
    numpy_array = sparse_matrix.toarray()

    print(" Step 3 : converting to a list for API call...")
    processed_review = numpy_array.tolist()

    print("\n Sending prediction request to the Vertex AI endpoint...")

    # Make the prediction
    response = endpoint.predict(instances=processed_review)

    # Print the prediction
    print(response)
    print(type(response))

'''