'''
Baseline classifier with Bag of Words implementation
'''
import Helper
import csv
import numpy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.cross_validation import KFold
from sklearn import metrics, cross_validation

resultFileName = "test0.txt"

#Using Logistic Regression, predicts categories for each article
def generatePredictions(corpus, targets):
    print("Generating Predictions...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    LogReg = LogisticRegression()
    '''
    CROSS VALIDATION: cv assigned folds global variable for number of requested folds
    '''
    kFold = KFold(len(targets), n_folds=Helper.folds, shuffle=True)
    scores = cross_val_score(LogReg, X, targets, cv=kFold)
    print("10-fold accuracy scores: ")
    print(scores)
    #This will do 10 folds, but each item is only predicted once
    predicted = cross_validation.cross_val_predict(LogReg, X, targets, cv=kFold)
    return predicted

class BagOfWords:
    filePath = sys.argv[1]
    corpus, targets, titles = Helper.createCorpus(filePath)
    predicted = generatePredictions(corpus, targets)
    Helper.printResults(targets, predicted, titles, resultFileName)
