import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# sample training data
texts = [
    "This is a positive text.",
    "I love this product!",
    "This movie is amazing.",
    "I don't like this at all.",
    "This is terrible."
]
labels = np.array([1, 1, 1, 0, 0])  # 1 for positive, 0 for negative

# tokenize the text and remove stop words
stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words=stop_words)

# convert the text into a matrix of word frequencies
word_counts = vectorizer.fit_transform(texts)

# train a Naive Bayes classifier on the training data
classifier = MultinomialNB()
classifier.fit(word_counts, labels)

# sample test data
test_texts = [
    "I hate this product!",
    "This movie is boring."
]

# preprocess the test data and convert to word frequencies
test_word_counts = vectorizer.transform(test_texts)

# classify the test data and print the results
predictions = classifier.predict(test_word_counts)
for i, text in enumerate(test_texts):
    print(f"{text} - {predictions[i]}")
