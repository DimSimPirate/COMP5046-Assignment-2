#TODO: map words to WordNet sysnets, mapping to similar documents from Wikipedia, explicit semantic analysis
#TODO: disambiguate named entities
#TODO: represent documents by latent topic signatures, Latent Dirichlet Allocation
#TODO: sentiment lexicon features

#Baseline classifier with Bag of Words implementation

import csv
import numpy
import collections
import re
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.cross_validation import KFold
from sklearn import metrics, datasets, svm, cross_validation
from collections import Counter, OrderedDict

folds = 10
printMetrics = True
printPredictions = True

#Filters the CSV file into an array of words for each article
def createCorpus(reader, includeBody):
    corpus = []
    categories = []
    titles = []
    for row in reader:
        text = ""
        if includeBody == True:
            text = row[2] + " " + row[3]
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
        with open('test0.txt', 'w') as f:
            for i in range(0, len(targets)):
                row = targets[i] + " " + predicted[i] + ": " + titles[i]
                f.write("%s\n" % str(row))

class BagOfWords:
    #Using Logistic Regression, predicts categories for each article
    def generatePredictions(corpus, targets):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        X = X.toarray()
        LogReg = LogisticRegression()
        '''
        CROSS VALIDATION: cv assigned folds global variable for number of requested folds
        '''
        kFold = KFold(len(targets), n_folds=folds, shuffle=True)

        scores = cross_val_score(LogReg, X, targets, cv=kFold)
        print("SCORES FOR EACH OF THE FOLDS")
        print(scores)
        predicted = cross_validation.cross_val_predict(LogReg, X, targets, cv=kFold)
        return predicted

    filePath = sys.argv[1]
    selectedFile = "";
    if filePath == "topic.csv":
        selectedFile = "topic"
    elif filePath == "virality.csv":
        selectedFile = "virality"
    else:
        selectedFile = "other"

    with open(filePath, "r+") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        corpus, targets, titles = createCorpus(reader, includeBody)
        predicted = generatePredictions(corpus, targets)
        printResults(targets, predicted, titles)

    csvfile.close()
