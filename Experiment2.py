#This experiment will extend the bag of words model by implementing n-grams

#TODO: words from only title, lead paragraph, all paragraphs
#TODO: clean text to reduce noise

import csv
import numpy
import collections
import re
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.cross_validation import KFold
from sklearn import metrics, datasets, svm, cross_validation
from collections import Counter, OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer

nGrams = 2
annotations = ["Business", "Entertainment", "Error", "Health", "Other", "Politics", "Science and Technology", "Society", "Sports", "War"]
#Sets whether the body is included in analysis, or if it is just the title
includeBody = False
#Sets the number of folds for validation
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
        with open('test2.txt', 'w') as f:
            for i in range(0, len(targets)):
                row = targets[i] + " " + predicted[i] + ": " + titles[i]
                f.write("%s\n" % str(row))
class nGrams:

    def generatePredictions(corpus, targets):
        vectorizer = CountVectorizer(ngram_range=(1, nGrams))
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
    nGrams = int(sys.argv[2])
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
        print("cerating corpus")
        corpus, targets, titles = createCorpus(reader, includeBody)
        print("Generating predictions")
        predicted = generatePredictions(corpus, targets)
        print("Evaluating results")
        printResults(targets, predicted, titles)

    csvfile.close()
