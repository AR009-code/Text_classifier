
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords

#wordDir=os.path.join(os.getcwd(),'sample words')
#df1= pd.read_csv('./positive-text.csv')
#df2= pd.read_csv('./negative-text.csv')

#__pwords=df1.values.tolist()
#__nwords=df2.values.tolist()
stops= set(('did', 'you', 'your', 'some', "you've", 'which', "isn", 'yourselves', 'having', "weren", 'just', 'her', 'those',
           'same', 'been', 'him', 'am', 'an', "hasn't", 'ourselves', 'has', 'up', 'out', 'only','into', "couldn't", "wouldn", "shouldn",
          'again', 'few', 'was', "needn't", "wasn't", 'while', 'this', 'these', 'she', 'he', 'himself', 'herself', 'under', 'so', 'can',
          'after', 'have', 'we', 'themselves', 'both', 'who', 'above', 'there', 'a', 'had', 'the', 'as', "you'll", 'yourself', "you'd",
          'll', 'where', 'with', 'most', 'than', 'before', 're', 'about', 'doing', 'that', 'theirs', 'their', 'are', 'during', 'being',
          'each', 'once', "she's", 'be', "mightn", 'and', 'should', 'when', 'what', 'down', 'yours', 'in', 'is', 'own', 'until',
          'to','its', 'it', 'more', 'now', "hadn", 'now', 'whome', 'myself', 'then', 'does', 'will', 'but', 'at', 'if', 'through',
          'they', 'me', "hasn", 'very', 'here', 'wasn', 'such', "you're", "should've", 'from', "hadn't", 'were', 'their', 'or', 'how', 'this',
          'ours', 'do', 'below', 'further', 'his', 'all', 'our', 'herself', 'why', 'aren', 'other', 'any', 'off','ain', "that'll", 
          'by', "it's", 'for', 'isn','itself', 'hers', 'them', 'between', 'on', 'because', 'too','my', 'of','i', 'always'))

#__training_words= list(__pwords + __nwords)

#words=[w[0] for w in __training_words]
#labels=[l[1] for l in __training_words]

class wordSamples:

    def __init__(self):
        pass

    #def getWords():
    #    return words

    #def getLabels():
    #    return labels

    def getStopWords():
        return stops