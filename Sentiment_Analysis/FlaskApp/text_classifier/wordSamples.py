
import pandas as pd

df1= pd.read_csv('./text_classifier/positive-words.csv')
df2= pd.read_csv('./text_classifier/negative-words.csv')

__pwords=df1.values.tolist()
__nwords=df2.values.tolist()
stop_words= set([
    'a', 'an', 'the', 'and', 'or', 'but', 'be', 'by', 'for', 'from', 'has', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'to', 'was', 'were', 'will', 'with' 
    ])

__training_words= list(__pwords + __nwords)

__words=[w[0] for w in __training_words]
__labels=[l[1] for l in __training_words]

class wordSamples:

    def __init__(self):
        pass

    def getWords(self):
        return __words

    def getLabels(self):
        return __labels

    def getStopWords(self):
        return stop_words