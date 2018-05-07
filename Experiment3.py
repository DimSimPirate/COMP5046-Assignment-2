#TODO: Use POS tagging as alternative representation
#Baseline classifier with Bag of Words implementation

import csv
import numpy
import collections
import re
import sys
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.cross_validation import KFold
from sklearn import metrics, datasets, svm, cross_validation
from collections import Counter, OrderedDict

#Sets whether the body is included in analysis, or if it is just the title
includeBody = False
#Sets the number of folds for validation
kFold = 2
printMetrics = True
printPredictions = False

#Filters the CSV file into an array of words for each article
def createCorpus(reader, includeBody):
    corpus = []
    categories = []
    titles = []
    for row in reader:
        text = ""
        if includeBody == True:
            text = row[3]
        else:
            text = row[2]
        titles.append(row[2])
        categories.append(row[0])
        corpus.append(text)
    targets = numpy.array(categories)
    return corpus, targets, titles

def printResults(targets, predicted, titles):
    if printMetrics == True:
        print(metrics.accuracy_score(targets, predicted))
        print(metrics.classification_report(targets, predicted))
    if printPredictions == True:
        for i in range(0, len(targets)):
            print(titles[i])
            print(targets[i] + " " + predicted[i])

class POSTagger:

    '''
    Function modified from https://nlp1000.wordpress.com/2016/12/19/pos-tagging-scikit-learn/
    '''
    def extract(text, index):
        token, tag = text[index]
        previousToken = ""
        previousTag = ""
        isNumber = False
        if index > 0:
            previousToken, previousTag = text[index - 1]
        try:
            if float(token):
                isNumber = True;
        except:
            pass
        features = {
            "token": token,
            "previousToken": previousToken,
            "suffix1": token[-1],
            "suffix2": token[-2:],
            "suffix3": token[-3:],
            "isNumber": isNumber
        }
        return features

    filePath = sys.argv[1]

    with open(filePath, "r+") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        print("Creating Corpus")
        corpus, targets, titles = createCorpus(reader, includeBody)
        POSCorpus = []
        print("Tagging Corpus")
        for i in range(0, len(corpus)):
            POSCorpus.append(nltk.pos_tag(nltk.word_tokenize(corpus[i])))

        print("Extracting features")
        textFeatures = []
        for text in POSCorpus:
            for i in range(len(text)):
                textFeatures.append(extract(text, i))

        #print("Generating classifications and scores")
        vectoriser = DictVectorizer()
        X = vectoriser.fit_transform(textFeatures)
        X = X.toarray()
        print(X.shape)
        #LogReg = LogisticRegression()
        #kFold = KFold(len(targets), shuffle=True, n_folds=kFold)

        #scores = cross_val_score(LogReg, X, targets, cv=kFold)
        #print("SCORES FOR EACH OF THE FOLDS")
        #print(scores)
        #predicted = cross_validation.cross_val_predict(LogReg, X, targets, cv=kFold)


    csvfile.close()
