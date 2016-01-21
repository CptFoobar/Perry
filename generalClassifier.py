import classificationPreprocessor as cp
import datasetRetriever as dr
import langid
import nltk
import sklearn
from functools import partial
from nltk.corpus import words
from sklearn.naive_bayes import GaussianNB

def trainGeneralClassifier():
    # DRY
    dr.init_all(200)
    # Get features, labelled bigrams, test and training sets
    dr.prepareDatasets()

    bigramFeatures, labelledBigrams = dr.getTrainingDataset()
    featureExtractor = partial(extractFeatures, gramFeatures = bigramFeatures)
    training_set = nltk.classify.apply_features(featureExtractor, labelledBigrams)
    labelledTest = dr.getTestDataset()
    test_set = nltk.classify.apply_features(featureExtractor, labelledTest)
    splitPoint = int(len(test_set) * 0.532)
    clf = trainNBClassifier(training_set + test_set[:splitPoint])
    printAcc(clf, test_set[splitPoint:])
    return clf

# Extract features from a bigram
def extractFeatures(doc, gramFeatures):
    document = set(doc)
    features = {}
    for gram in gramFeatures:
        features['contains(({0}, {1}))'.format(gram[0], gram[1])] = (gram in document)
    return features


# Naive Bayes Classifier - nltk implementation - using manual IDF
def trainNBClassifier(training_set):
    return nltk.NaiveBayesClassifier.train(training_set)


def printAcc(classifier, test_set):
    print "%.2f%%" % (nltk.classify.accuracy(classifier, test_set) * 100)


def predict(tweet, clf):

    PREDICTIONS = {
        '0' : "Definitely not sarcastic",
        '1' : "Probably not sarcastic",
        '2' : "Can't say",
        '3' : "Probably sarcastic",
        '4' : "Definitely sarcastic"
    }

    tweetBigrams = dr.getBigrams(tweet)
    bigramFeatures, _ = dr.getTrainingDataset()
    featureExtractor = partial(extractFeatures, gramFeatures = bigramFeatures)
    if tweetBigrams is not None:
        pr = clf.classify(featureExtractor(tweetBigrams))
        return PREDICTIONS[str(pr)]


# Extract features from a bigram
def extractFeatures(doc, gramFeatures):
    document = set(doc)
    features = {}
    for gram in gramFeatures:
        features['contains(({0}, {1}))'.format(gram[0], gram[1])] = (gram in document)
    return features
