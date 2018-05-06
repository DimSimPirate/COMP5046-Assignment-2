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
from sklearn import metrics, datasets, svm, cross_validation
from collections import Counter, OrderedDict

annotations = ["Business", "Entertainment", "Error", "Health", "Other", "Politics", "Science and Technology", "Society", "Sports", "War"]
#Sets whether the body is included in analysis, or if it is just the title
includeBody = True
#Sets the number of folds for validation
folds = 2
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
            text = row[2] + " " + row[3]
        else:
            text = row[2]
        titles.append(row[2])
        categories.append(row[0])
        corpus.append(text)
    targets = numpy.array(categories)
    return corpus, targets, titles

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
        predicted = cross_validation.cross_val_predict(LogReg, X, targets, cv=folds)
        return predicted

    def printResults(targets, predicted, titles):
        if printMetrics == True:
            print(metrics.accuracy_score(targets, predicted))
            print(metrics.classification_report(targets, predicted))
        if printPredictions == True:
            for i in range(0, len(targets)):
                print(titles[i])
                print(targets[i] + " " + predicted[i])

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
