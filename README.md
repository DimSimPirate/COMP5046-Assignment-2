Install Anaconda for simplest solution as all utilised packages are included in the distribution

REQUIRED PACKAGES:
    numpy
    sklearn
    nltk

Before running experiment 3
    type the following in the command line:
        py
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')

To run each experiment, from the command line enter the following

    py experiment0.py 'data.csv'
    py experiment1.py 'data.csv'
    py experiment2.py 'data.csv' 'nGrams'

    data.csv refers to the text file to evaluate
    nGrams is an integer for the number of requested n-grams
