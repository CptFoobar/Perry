import classificationPreprocessor as cp
import datasetRetriever as dr
import langid
import nltk
import sklearn
from functools import partial
from nltk.corpus import words
from sklearn.naive_bayes import GaussianNB

clf = None

def trainGeneralClassifier():
    global clf

    # DRY
    dr.init_all(200)
    # Get features, labelled bigrams, test and training sets
    # dr.prepareDatasets()

    bigramFeatures, labelledBigrams = dr.getTrainingDataset()
    featureExtractor = partial(extractFeatures, gramFeatures = bigramFeatures)
    training_set = nltk.classify.apply_features(featureExtractor, labelledBigrams)
    labelledTest = dr.getTestDataset()
    test_set = nltk.classify.apply_features(featureExtractor, labelledTest)
    splitPoint = int(len(test_set) * 0.8)
    clf = trainNBClassifier(training_set + test_set[:splitPoint])
    clf.classify(test_set[0][0])

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
    print nltk.classify.accuracy(classifier, test_set)


# Transform tweet to vector
def vectorizeTweet(tweet):
    tweet = ' '.join(tweet.split())
    lang, conf = langid.classify(tweet)
    if lang == 'en' and conf > 0.99:
        goodTweet, tweet = cp.processTweet(tweet)
        tweet = tweet.tweet
        if goodTweet:
            tweet = dr.cleanTweet(tweet, words.words())
            if len(tweet) > 0:
                tweet = dr.vectorizeOne(tweet)
                return tweet


def predict(tweet):
    global clf
    vecTweet = vectorizeTweet(tweet)
    if vecTweet is not None:
        return clf.classify(vecTweet)
