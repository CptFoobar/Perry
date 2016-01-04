import datasetRetriever as dr
import nltk
import numpy as np
import sklearn
import time
from functools import partial
from sklearn.cluster import KMeans, Birch
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as DTC
from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier
from tweetTemplate import *

def main():
    t0 = time.time()
    # DRY
    # dr.init(200)
    # Get features, labelled bigrams, test and training sets
    dr.prepareDatasets()
    bigramFeatures, labelledBigrams = dr.getTrainingDataset()
    featureExtractor = partial(extractFeatures, gramFeatures = bigramFeatures)
    train_features, train_labels = dr.getTrainingSet()
    test_features, test_labels = dr.getTestSet()
    training_set = nltk.classify.apply_features(featureExtractor, labelledBigrams)
    labelledTest = dr.getTestDataset()
    test_set = nltk.classify.apply_features(featureExtractor, labelledTest)

    for i in range(1,4):
        classifyKNN_sklearn((train_features, train_labels), (test_features, test_labels), i)

    classifyNB_sklearn((train_features, train_labels), (test_features, test_labels))

    mClassifyNB(training_set, test_set)
    mClassifyNB_sklearn(training_set, test_set)
    for i in range(1, 4):
        mClassifyKNN(training_set, test_set, i)

    ld_train = dr.getLabelledDataset("../data/training_set.tds")
    ld_test = dr.getLabelledDataset("../data/test_set.tds")

    classifyDT_sklearn((train_features, train_labels), (test_features, test_labels), 8)
    classifyKMC_sklearn((train_features, train_labels), (test_features, test_labels))

    classifyNB_tb(ld_train, ld_test)
    classifyDT_tb(ld_train, ld_test)

    print "Total time of execution: {0} mins.".format(str(int((time.time() - t0) / 60)))

    return


# Naive Bayes Classifier - nltk implementation - using manual IDF
def mClassifyNB(training_set, test_set):
    print "\nNaive Bayes Classifier"
    print "Training..."
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print "Training complete."

    print "Accuracy of Naive Bayes Classifier on Training set is " + \
                str(nltk.classify.accuracy(classifier, training_set))
    print "Accuracy of Naive Bayes Classifier on Test set is " + \
                str(nltk.classify.accuracy(classifier, test_set))


# k Nearest Neighbors - sklearn implementation
def classifyKNN_sklearn(training_set, test_set, n):
    print "\nK-Nearest Neighbor Classifier with {0} nearest neighbors.".format(str(n))
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    classifier = KNeighborsClassifier(n_neighbors=n)
    print "Training..."
    classifier.fit(train_features, train_labels)
    print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    te_acc = accuracy_score(classifier.predict(test_features.todense()), test_labels)
    print "Accuracy of KNN Classifier on Training set is " + \
            str(tr_acc)
    print "Accuracy of KNN Classifier on Test set is " + \
            str(te_acc)


# Naive Bayes classifier - sklearn implementation
def classifyNB_sklearn(training_set, test_set):
    print "\nNaive Bayes (sklearn) Classifier"
    classifier = GaussianNB()
    print "Training."
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    classifier.fit(train_features, train_labels)
    print "Training complete."
    print "Accuracy of Naive Bayes (sklearn) Classifier on Training set is " + \
            str(accuracy_score(classifier.predict(train_features), train_labels))
    print "Accuracy of Naive Bayes (sklearn) Classifier on Test set is " + \
            str(accuracy_score(classifier.predict(test_features.todense()), test_labels))


# Decision Trees - sklearn implementation
def classifyDT_sklearn(training_set, test_set, i):
    print "\nDecision trees (sklearn) Classifier ({0})".format(str(i))
    print "Training."
    # Decision Trees: min-split-sample = 8 and max-features = 200
    classifier = DTC(min_samples_split=3)
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    classifier.fit(train_features, train_labels)
    #print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    te_acc = accuracy_score(classifier.predict(test_features), test_labels)

    print "Accuracy of KNN Classifier on Training set is " + \
            str(tr_acc)
    print "Accuracy of KNN Classifier on Test set is " + \
            str(te_acc)


# K Means Clustering classifier - sklearn implementation
def classifyKMC_sklearn(training_set, test_set):
    print "\nK Means Clustering (sklearn) Classifier"
    classifier = KMeans(n_clusters=5)
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    print "Training."
    classifier.fit(train_features)
    print "Training complete."
    print "Accurracy of KMC train: " + calcAcc(train_features, train_labels, classifier)
    print "Accurracy of KMC test: " + calcAcc(test_features, test_labels, classifier)


# Naive Bayes Classifier - TextBlob implementation
def classifyNB_tb(training_set, test_set):
    print "\nTextBlob Naive Bayes."
    cl = NaiveBayesClassifier(training_set)
    print "Trained"
    print "Train set accuracy: " + str(cl.accuracy(training_set))
    print "Test set accuracy: " + str(cl.accuracy(test_set))


# Decision trees - TextBlob implementation
def classifyDT_tb(training_set, test_set):
    print "\nTextBlob Decision Trees."
    cl = DecisionTreeClassifier(training_set)
    print "Trained"
    print "Train set accuracy: " + str(cl.accuracy(training_set))
    print "Test set accuracy: " + str(cl.accuracy(test_set))


# Naive Bayes Classifier (sklearn implementation) - using manual IDF
def mClassifyNB_sklearn(training_set, test_set):
    print "\nNaive Bayes (sklearn) Classifier (m)"
    classifier = GaussianNB()
    train_features, train_labels, test_features, test_labels = \
                                getFeaturesLabels(training_set, test_set)

    print "Training."
    classifier.fit(train_features, train_labels)
    print "Training complete."
    print "Accuracy of Naive Bayes (sklearn) Classifier on Training set is " + \
            str(accuracy_score(classifier.predict(train_features), train_labels))
    print "Accuracy of Naive Bayes (sklearn) Classifier on Test set is " + \
            str(accuracy_score(classifier.predict(test_features), test_labels))


# k Nearest Neighbors - using manual IDF
def mClassifyKNN(training_set, test_set, n):
    print "\nK-Nearest Neighbor Classifier with {0} nearest neighbors. (m)".format(str(n))
    train_features, train_labels, test_features, test_labels = \
                                getFeaturesLabels(training_set, test_set)

    classifier = KNeighborsClassifier(n_neighbors=n)
    print "Training..."
    classifier.fit(train_features, train_labels)
    print "Training complete."
    print "Accuracy of KNN Classifier on Training set is " + \
            str(accuracy_score(classifier.predict(train_features), train_labels))
    print "Accuracy of KNN Classifier on Test set is " + \
            str(accuracy_score(classifier.predict(test_features), test_labels))


# Extract features from a bigram
def extractFeatures(doc, gramFeatures):
    document = set(doc)
    features = {}
    for gram in gramFeatures:
        features['contains(({0}, {1}))'.format(gram[0], gram[1])] = (gram in document)
    return features


# Extract features and labels from training ans test sets, given a labellled set
def getFeaturesLabels(training_set, test_set):
    train_features = []
    train_labels = []
    for features, label in training_set:
        tf = []
        for feature in features:
            if features[feature]:
                tf.append(1.)
            else: tf.append(0.)
        train_features.append(tf)
        train_labels.append(label)
    test_features = []
    test_labels = []
    for feature, label in test_set:
        tf = []
        for feature in features:
            if features[feature]:
                tf.append(1.)
            else: tf.append(0.)
        test_features.append(tf)
        test_labels.append(label)

    return train_features, train_labels, test_features, test_labels


# K Means classifier returns an array as a prediction, and doesn't work properly
# with accuracy_score. Hence, a util
def calcAcc(features, labels, classifier):
    j = 0.
    for i in range(len(labels)):
        p = str(classifier.predict(features[i])[0])
        l = labels[i]
        if p == l:
            j += 1.
    return str(float(j / len(labels)))


if __name__ == "__main__":
    main()
