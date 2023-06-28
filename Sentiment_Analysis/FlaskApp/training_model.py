
import re
import numpy as np
from wordSamples import wordSamples as ws


# sample training data
texts = [
   "The speech aroused the enthusiasm of the students.",
   "I am hating this movies",
   "I love this movies",
   "I loving this song",
   "you are a good boy",
   "People do not like me.",
   "People loves me a lot",
   "I know you like me so much",
   "People always like me",
   "I like your inspiraation"
]

labels = np.array([1, 0, 1, 1,1, 0, 1, 1, 1, 1 ])  # 1 for positive, 0 for negative
 
#print("sample text: \n{0}".format(texts))
print("The defined stop words :", ws.getStopWords())

# define a function to preprocess the text
def preprocess_text(text):
    # convert to lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # remove digits
    text = re.sub(r'\d+', '', text)
    # remove stopwords (optional)
    stopwords = ws.getStopWords()
    words = text.split()
    words = [word for word in words if word not in stopwords]
    # return preprocessed text as a string
    return ' '.join(words)

# preprocess the text and count word frequencies
word_counts = []
for text in texts:
    text = preprocess_text(text)
    words = text.split()
    word_counts.append({word: words.count(word) for word in words})

print("word counts:", word_counts)

# create a dictionary of all unique words in the training data
all_words = set()
for word_count in word_counts:
    all_words.update(word_count.keys())
print("all words:", all_words)
word_to_index = {word: i for i, word in enumerate(all_words)}

print("word to index: \n", word_to_index)
# convert the text into a matrix of word frequencies
num_texts = len(texts)
num_words = len(all_words)
matrix = np.zeros((num_texts, num_words))

for i, word_count in enumerate(word_counts):
    for word, count in word_count.items():
        j = word_to_index[word]
        matrix[i, j] = count
 
print("updated matrix: \n", matrix)
# train a Naive Bayes classifier on the training data
class_prior = np.bincount(labels) / len(labels)
print("occurenses: ", np.bincount(labels))
#print( "matrix label ==0 \n",matrix[labels == 0].sum(axis=0))

def word_classifier():
    word_counts_positive = matrix[labels == 1].sum(axis=0)
    print("positive count:", word_counts_positive)
    word_counts_negative = matrix[labels == 0].sum(axis=0)
    print("negative count:", word_counts_negative)
    total_counts = matrix.sum(axis=0)
    print("total counts: ", total_counts)
    return (word_counts_positive, word_counts_negative, total_counts)


def p_word_pos_and_neg(wcp, wcn, tc):
    p_word_positive = (wcp + 1) / (tc + num_words)
    print("P_word_positive: ", p_word_positive)
    p_word_negative = (wcn + 1) / (tc + num_words)
    print("P_word_negative: ", p_word_negative)
    return (p_word_positive, p_word_negative)

(word_counts_positive, word_counts_negative, total_counts)= word_classifier()

(p_word_positive, p_word_negative)= p_word_pos_and_neg ( word_counts_positive, word_counts_negative, total_counts)

log_p_positive = np.log(class_prior[1]) + np.sum(np.log(p_word_positive) * matrix, axis=1)
print("log_p_positive: ", log_p_positive)
log_p_negative = np.log(class_prior[0]) + np.sum(np.log(p_word_negative) * matrix, axis=1)
print("log_p_negative: ", log_p_negative)
predictions = (log_p_positive > log_p_negative).astype(int)
print("predictions: ",np.log(p_word_positive))


