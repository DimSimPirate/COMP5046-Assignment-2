#This experiment will clean up some of the textual data and apply TF-IDF weighting
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
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

        #TODO: normalise numbers

# Dictionary modified from https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}
#NOTE: this does not take into consideration ambiguous contractions

annotations = ["Business", "Entertainment", "Error", "Health", "Other", "Politics", "Science and Technology", "Society", "Sports", "War"]
#Sets whether the body is included in analysis, or if it is just the title
includeBody = True
#Sets the number of folds for validation
folds = 10
printMetrics = True
printPredictions = False
#Removes words that only appear once in the entire corpus
minCutoff = 1;
#Float to exclude the most occuring words
maxCutoff = 0.9
nGrams = 2

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

#Function to expand contractions in the corpus
compiledContractions = re.compile('(%s)' % '|'.join(contractions.keys()))
def expandCont(paragraph):
    def replace(match):
        return contractions[match.group(0)]
    return compiledContractions.sub(replace, paragraph.lower())

class normAndWeights:

    def predictions(corpus, targets):
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=minCutoff, max_df=maxCutoff, norm='l2', ngram_range=(1, nGrams))
        X = vectorizer.fit_transform(corpus)
        X = X.toarray()
        LogReg = LogisticRegression()
        '''
        CROSS VALIDATION: cv assigned folds global variable for number of requested folds
        '''
        kFold = KFold(len(targets), shuffle=True, n_folds=folds)
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

        for i in range(0, len(corpus)):
            replacement = expandCont(corpus[i])
            corpus[i] = replacement

        predicted = predictions(corpus, targets)
        printResults(targets, predicted, titles)

    csvfile.close()
