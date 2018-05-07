'''
This script contains function definitions and global variables that are used by all experiments
'''

import csv
import numpy
import sklearn.metrics as metrics

#If true the body text is classified, if false just the title
includeBody = False
#Sets the number of folds for validation
folds = 10
#Whether or not to print metrics
printMetrics = True
#Whether or not to write results to file
printPredictions = True

#Extracts required information from csv file
def createCorpus(filePath):
    print("Extracting info from data file...")
    with open(filePath, "r+") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

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

    csvfile.close()

#Prints the accuracy score, classification report, and writes all classifications to a file
def printResults(targets, predicted, titles, name):
    if printMetrics == True:
        print("Accuracy score: ")
        print(metrics.accuracy_score(targets, predicted))
        print("Additional metrics: ")
        print(metrics.classification_report(targets, predicted))
    if printPredictions == True:
        with open(name, 'w') as f:
            f.write("GoldClass, Predicted, Title\n")
            for i in range(0, len(targets)):
                row = targets[i] + ", " + predicted[i] + ", " + titles[i]
                f.write("%s\n" % str(row))
        f.close()
