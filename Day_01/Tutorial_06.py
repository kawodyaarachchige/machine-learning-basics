# Importing the required libraries
# wikipediaapi → to get information from Wikipedia
# nltk → for Natural Language Processing (works with text)
# ssl → to handle secure internet connections
# re → for regular expressions (used for searching/replacing text patterns)
# numpy → a library for math calculations
# stopwords, WordNetLemmatizer → for text cleaning
# TfidfVectorizer → for finding important words in text
# KMeans → for grouping similar words or topics
import wikipediaapi
import nltk
import ssl
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- Fixing an Internet certificate issue ---
# Sometimes, downloading NLTK data fails because of SSL (security) problems.
# The below code ignores certificate errors so downloads can work.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # If your Python version doesn't have this feature, do nothing.
    pass
else:
    # If it exists, use it to avoid SSL errors.
    ssl._create_default_https_context = _create_unverified_https_context

# --- Downloading NLTK data ---
# These lines are commented out because you may have already downloaded the data.
# You can remove the "#" if you need to run them again.
# nltk.download('stopwords')  # List of common words like "is", "the", "on" that we usually remove
# nltk.download('wordnet')    # Database to find base form of words (like "running" → "run")
# nltk.download('omw-1.4')    # Word meanings in many languages

print("NLTK data downloaded successfully using the manual method.")

'''

# --- Setting up Wikipedia API ---
# 'test_pr/1.0' is just a name for our program
# 'en' means we want English Wikipedia
wiki_api = wikipediaapi.Wikipedia('test_pr/1.0', 'en')

# --- Getting a Wikipedia page ---
# Let's try to find the Wikipedia page for "Kusal Gunasekara"
page = wiki_api.page("Kusal Gunasekara")

# --- Checking if the page exists ---
if page.exists():
    # Print the title and the full text of the page
    print("Title : ", page.title, "\n")
    print("Text : ", page.text)
else:
    # If the page is not found
    print("Page does not exist")

'''


# --- Step 1: Choose some Wikipedia article titles ---
article_titles = [
    "Galaxy", "Black hole", "Supernova",  # Space topics
    "DNA", "Photosynthesis", "Evolution", # Biology topics
    "Machine Learning", "Artificial intelligence", "Computer programming"  # Technology topics
]

# --- Step 2: Fetch article text from Wikipedia ---
document = []  # This will store the text of all articles
wiki_api = wikipediaapi.Wikipedia("MyClusteringProject/1.0", "en")  # Our Wikipedia connection

for title in article_titles:
    page = wiki_api.page(title)  # Get the page
    if page.exists():
        document.append(page.text)  # Store the page text
    else:
        print(f"Page '{title}' does not exist")

print(f"Fetched {len(document)} articles from Wikipedia.")

# --- Step 3: Prepare tools for text cleaning ---
stop_words = set(stopwords.words('english'))  # Common words to ignore
lemmatizer = WordNetLemmatizer()              # To get base forms of words

# --- Step 4: Define a function to clean text ---
def preprocess_text(text):
    text = text.lower()  # Make everything lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers, punctuation, special symbols
    words = text.split()  # Split text into words
    processed_words = [
        lemmatizer.lemmatize(word)  # Change word to base form
        for word in words
        if word not in stop_words  # Remove common useless words
    ]
    return ' '.join(processed_words)  # Put cleaned words back into a sentence

# --- Step 5: Clean all the articles ---
processed_documents = [preprocess_text(doc) for doc in document]

# --- Step 6: Convert text into numbers using TF-IDF ---
vectorizer = TfidfVectorizer(max_features=1000)  # Only keep top 1000 important words
tfidf_matrix = vectorizer.fit_transform(processed_documents)

print("TF-IDF matrix created successfully.")
print(f"Shape of the matrix: {tfidf_matrix.shape}")  # (documents, words)

# --- Step 7: Apply KMeans clustering ---
k = 3  # Number of groups (clusters) we want
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)

print("Clustering completed successfully.")

# --- Step 8: Show which cluster each article belongs to ---
cluster_labels = kmeans.labels_
print("Cluster labels for each article:")
# for title, label in zip(article_titles, cluster_labels):
#     print(f"{title} → Cluster {label}")
print(cluster_labels)

new_doc = " An algorithm is a set of well-defined instructions designed to perform a specific task or solve a computational problem. In computer science, the study of algorithms is fundamental to creating efficient and scalable software. Data structures, such as arrays and hash tables, are used to organize data in a way that allows these algorithms to access and manipulate it effectively."

new_doc_processed = preprocess_text(new_doc)
new_doc_vector = vectorizer.transform([new_doc_processed])
new_doc_cluster = kmeans.predict(new_doc_vector)[0]

print("New document belongs to cluster", new_doc_cluster)