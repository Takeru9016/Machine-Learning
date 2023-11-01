import pickle

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords data
nltk.download('stopwords')
nltk.download('punkt')

# Load data from your CSV file
data = pd.read_csv('movie.csv')

# Extract movie reviews and labels
reviews = data['text']
labels = data['label']

# Preprocessing function
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Preprocess the reviews
reviews = [preprocess_text(review) for review in reviews]

# TF-IDF vectorization
# You can adjust the number of features as needed
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(reviews)

# Build and train the sentiment analysis model
model = LogisticRegression()
model.fit(X, labels)

# Model evaluation (optional)
y_pred = model.predict(X)
accuracy = accuracy_score(labels, y_pred)
print("Accuracy:", accuracy)
print(classification_report(labels, y_pred))

# Make predictions for new reviews
while True:
    new_review = input("Enter a movie review or type 'exit' to quit: ")

    if new_review.lower() == 'exit':
        break

    new_review = preprocess_text(new_review)
    new_review_vectorized = tfidf_vectorizer.transform([new_review])
    prediction = model.predict(new_review_vectorized)

    if prediction[0] == 1:
        print("Predicted Sentiment: Positive")
    else:
        print("Predicted Sentiment: Negative")

# You can save the trained model for later use if needed
pickle.dump(model, open('./sentiment_model.p', 'wb'))
