#This experiment will extend the bag of words model by implementing n-grams

#TODO: words from only title, lead paragraph, all paragraphs
#TODO: clean text to reduce noise

import Experiment0

n = 2

class nGrams:

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
        corpus, targets, titles = createCorpus(reader)
        vectorizer = CountVectorizer(ngram_range=(1, n))
