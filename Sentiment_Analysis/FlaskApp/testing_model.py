import re
import numpy as np
import text_classifier.training_model as train_m
# sample test data
#test_text = [
#    "I love this product!",
#    "This movie is boring.",
#    "I liked that you had appreciated my gameplay.",
#    "There will be less chance for the accomplishment",
#]

# preprocess the test data and convert to word frequencies

def run_model(test_text):
    test_word_counts = []
    result=[]
    for text in test_text:
        text = train_m.preprocess_text(text)
        words = text.split()
        test_word_counts.append({word: words.count(word) for word in words})

    test_matrix = np.zeros((len(test_text), train_m.num_words))
    for i, word_count in enumerate(test_word_counts):
        for word, count in word_count.items():
            if word in train_m.word_to_index:
                j = train_m.word_to_index[word]
                test_matrix[i, j] = count

    # classify the test data and print the results
    test_log_p_positive = np.sum(np.log(train_m.p_word_positive) * test_matrix, axis=1)
    test_log_p_negative = np.sum(np.log(train_m.p_word_negative) * test_matrix, axis=1)
    test_predictions = (test_log_p_positive > test_log_p_negative).astype(int)
    for i, text in enumerate(test_text):
        result.append((text,test_predictions))
        print(f"{text} - {test_predictions[i]}")
    return result

#if __name__ == '__main__':
#    print(run_model(["I love this animal"])[0][1])

