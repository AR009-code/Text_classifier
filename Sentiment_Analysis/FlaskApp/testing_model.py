from multiprocessing.reduction import duplicate
import re
import numpy as np
import training_model as train_m
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
    
    print("///testing model///")

    test_matrix = np.zeros((len(test_text), train_m.num_words))
  
    for i, word_count in enumerate(test_word_counts):
        for word, count in word_count.items():
            if word in train_m.word_to_index:
                j = train_m.word_to_index[word]
                test_matrix[i, j] = count

    print("test_words: ", test_word_counts)
    print("test_matrix: ", test_matrix)
    (train_m.p_word_positive,train_m.p_word_negative)= train_m.p_word_pos_and_neg ( train_m.word_counts_positive, train_m.word_counts_negative, train_m.total_counts)
    duplicate_pn_index= [j for j, freq in enumerate(train_m.total_counts) if(freq > 1.0) ]  # list out the indices of the duplicate words with both +/- sentiment
    print("duplicate pn words: ", duplicate_pn_index)

    m_word_counts_positive= train_m.word_counts_positive
    m_word_counts_negative= train_m.word_counts_negative

    if(len(duplicate_pn_index)):    #it will execute only when the model have duplicate words for both positive and negative sentiment
        
        for d in test_word_counts:
            test_word_index=[train_m.word_to_index.get(x) for x in d.keys() if train_m.word_to_index.get(x) != None]
            print("text word index: ", test_word_index)
            freq_count= len(d)
            print("freq_count:", freq_count)
            feature_extract=[x for x in test_word_index if(x not in duplicate_pn_index)]
            print("feature: ", feature_extract)
            for x in feature_extract:
                for i in test_word_index:          
                    if(train_m.word_counts_positive[x]!=0.0):
                        m_word_counts_positive[i]=freq_count
                    else:
                        m_word_counts_negative[i]=freq_count

        m_total_counts=np.add(m_word_counts_positive, m_word_counts_negative) 
        #here modifying the values of the pre-naive bayes trained results on the basis of the test results
        train_m.p_word_positive, train_m.p_word_negative= train_m.p_word_pos_and_neg(m_word_counts_positive, m_word_counts_negative, m_total_counts)
    

    print("updated test matrix: ", test_matrix)
    # classify the test data and print the results
    # the train_m.p_word_positive and train_m.p_word_negative can be a modified values or may using the pre trained values
    test_log_p_positive = np.sum(np.log(train_m.p_word_positive) * test_matrix, axis=1)
    print("test_log_p_positive: ",test_log_p_positive)
    test_log_p_negative = np.sum(np.log(train_m.p_word_negative) * test_matrix, axis=1)
    print("test_log_p_negative: ",test_log_p_negative)
    test_predictions = (test_log_p_positive > test_log_p_negative).astype(int)
    for i, text in enumerate(test_text):
        result.append((text,test_predictions))
        print(f"{text} - {test_predictions[i]}")
    return (result, m_word_counts_positive, m_word_counts_negative, test_word_counts, test_matrix, test_log_p_positive, test_log_p_negative)

#if __name__ == '__main__':
#    print(run_model(["I love this animal"])[0][1])

