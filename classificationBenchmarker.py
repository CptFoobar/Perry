import datasetRetriever as dr
import result_visualizer as rv
from json2html import *
import json
import nltk
import numpy as np
import sklearn
import sys
import time
from functools import partial
from nltk.metrics.scores import f_measure, recall, precision
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier
from tweetTemplate import *

def main():

    t0 = time.time()
    # DRY
    dr.init(200)
    results = []
    rmax = 1    # Intended for k-fold cross validation. Vary as k
    for r in range(0, rmax):
        sys.stdout.write("\rOverall Progress: %d%%" % int(r * 100 / rmax))
        sys.stdout.flush()
        print
        # Get features, labelled bigrams, test and training sets
        dr.prepareDatasets(0.78)
        bigramFeatures, labelledBigrams = dr.getTrainingDataset()
        featureExtractor = partial(extractFeatures, gramFeatures = bigramFeatures)
        train_features, train_labels = dr.getTrainingSet()
        test_features, test_labels = dr.getTestSet()

        training_set = nltk.classify.apply_features(featureExtractor, labelledBigrams)
        labelledTest = dr.getTestDataset()
        test_set = nltk.classify.apply_features(featureExtractor, labelledTest)
        print "training_set: {0}, test_set: {1}".format(str(len(training_set)), str(len(test_set)))

        '''
        for i in range(1,4):
            res = classifyKNN_sklearn((train_features, train_labels), (test_features, test_labels), i)
            results.append(res)
        '''
        '''
        res = classifyNB_sklearn((train_features, train_labels), (test_features, test_labels))
        results.append(res)
        '''
        results.append(mClassifyNB(training_set, test_set))
        results.append(mClassifyNB_sklearn(training_set, test_set))
        '''
        for i in range(1, 4):
            results.append(mClassifyKNN(training_set, test_set, i))

        results.append(classifyDT_sklearn((train_features, train_labels), (test_features, test_labels), 8))
        results.append(classifyKMC_sklearn((train_features, train_labels), (test_features, test_labels)))
        results.append(classifySVM_sklearn((train_features, train_labels), (test_features, test_labels)))
        '''

    resultsFile = open("results.json", 'w')
    json.dump(list(results), resultsFile)
    resultsFile.close()

    ld_train = dr.getLabelledDataset("../data/training_set.tds")
    ld_test = dr.getLabelledDataset("../data/test_set.tds")
    # Text blob is know to outperform current implementation, leave it out for now
    #classifyNB_tb(ld_train, ld_test)
    #classifyDT_tb(ld_train, ld_test)
    htmlTemplateBegin = "<!DOCTYPE html><html><head><title>Sarcastic results</title></head><body><h2>Sarcasm Detection Results</h2>"
    htmlTemplateEnd = "</body></html>"
    resHtml = open("Benchmark.html", 'w')
    resHtml.write(htmlTemplateBegin)
    resHtml.write(json2html.convert(json={'results': results}))
    resHtml.write(htmlTemplateEnd)
    resHtml.close()

    print "Execution complete."
    print "Total time of execution: {0} mins.".format(str(int((time.time() - t0) / 60)))
    # rv.visualize("results.json")
    return


# Naive Bayes Classifier - nltk implementation - using manual IDF
def mClassifyNB(training_set, test_set):
    print "Naive Bayes Classifier"
    # print "Training..."
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # print "Training complete."
    tr_acc = nltk.classify.accuracy(classifier, training_set)
    te_acc = nltk.classify.accuracy(classifier, test_set)
    results = classifier.classify_many([fs for (fs, lbl) in test_set])
    reference = [lbl for (fs, lbl) in test_set]
    f1_score = f_measure(set(reference), set(results))
    p = precision(set(reference), set(results))
    r = recall(set(reference), set(results))
    print "NLTK NB - manual IDF: p = " + str(p) + " r = " + str(r)
    return generateJson("Naive Bayes - manual IDF", tr_acc, te_acc, f1_score)


# k Nearest Neighbors - sklearn implementation
def classifyKNN_sklearn(training_set, test_set, n):
    print "K-Nearest Neighbor Classifier with {0} nearest neighbors.".format(str(n))
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    classifier = KNeighborsClassifier(n_neighbors=n)
    # print "Training..."
    classifier.fit(train_features, train_labels)
    # print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    te_acc = accuracy_score(classifier.predict(test_features.todense()), test_labels)
    f_score = f1_score(test_labels, classifier.predict(test_features.todense()), average='binary')
    return generateJson("k-NN, k = %d" % n, tr_acc, te_acc, f_score)


# Naive Bayes classifier - sklearn implementation
def classifyNB_sklearn(training_set, test_set):
    print "Naive Bayes (sklearn) Classifier"
    classifier = GaussianNB()
    # print "Training."
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    classifier.fit(train_features, train_labels)
    # print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    predictions = classifier.predict(test_features.todense())
    te_acc = accuracy_score(predictions, test_labels)
    f_score = f1_score(test_labels, predictions, average='binary')

    return generateJson("Naive Bayes (sklearn)", tr_acc, te_acc, f_score)


def classifySVM_sklearn(training_set, test_set):
    print "SVM (sklearn) Classifier"
    classifier = SVC(C=1000.0)
    # print "Training."
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    classifier.fit(train_features, train_labels)
    # print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    te_acc = accuracy_score(classifier.predict(test_features.todense()), test_labels)
    f_score = f1_score(test_labels, classifier.predict(test_features.todense()), average='binary')

    return generateJson("SVM (sklearn)", tr_acc, te_acc, f_score)


# Decision Trees - sklearn implementation
def classifyDT_sklearn(training_set, test_set, i):
    print "Decision trees (sklearn) Classifier ({0})".format(str(i))
    print "Training."
    # Decision Trees: min-split-sample = 8 and max-features = 200
    classifier = DTC(min_samples_split=3)
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    classifier.fit(train_features, train_labels)
    #print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    te_acc = accuracy_score(classifier.predict(test_features), test_labels)
    f_score = f1_score(test_labels, classifier.predict(test_features.todense()), average='binary')

    return generateJson("Decision Trees (sklearn)", tr_acc, te_acc, f_score)


# K Means Clustering classifier - sklearn implementation
def classifyKMC_sklearn(training_set, test_set):
    print "K Means Clustering (sklearn) Classifier"
    classifier = KMeans(n_clusters=5)
    train_features, train_labels = training_set
    test_features, test_labels = test_set
    # print "Training."
    classifier.fit(train_features)
    # print "Training complete."
    tr_acc = calcAcc(train_features, train_labels, classifier)
    te_acc = calcAcc(test_features, test_labels, classifier)
    f_score = f1_score(test_labels, classifier.predict(test_features.todense()), average='weighted')

    return generateJson("K Means Clustering (sklearn)", tr_acc, te_acc, f_score)


# Naive Bayes Classifier - TextBlob implementation
def classifyNB_tb(training_set, test_set):
    print "TextBlob Naive Bayes."
    cl = NaiveBayesClassifier(training_set)
    print "Trained"
    print "Train set accuracy: " + str(cl.accuracy(training_set))
    print "Test set accuracy: " + str(cl.accuracy(test_set))


# Decision trees - TextBlob implementation
def classifyDT_tb(training_set, test_set):
    print "TextBlob Decision Trees."
    cl = DecisionTreeClassifier(training_set)
    print "Trained"
    print "Train set accuracy: " + str(cl.accuracy(training_set))
    print "Test set accuracy: " + str(cl.accuracy(test_set))


# Naive Bayes Classifier (sklearn implementation) - using manual IDF
def mClassifyNB_sklearn(training_set, test_set):
    print "Naive Bayes (sklearn) Classifier (m)"
    classifier = GaussianNB()
    train_features, train_labels, test_features, test_labels = \
                                getFeaturesLabels(training_set, test_set)

    # print "Training."
    classifier.fit(train_features, train_labels)
    # print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    te_acc = accuracy_score(classifier.predict(test_features), test_labels)
    f_score = f1_score(test_labels, classifier.predict(test_features), average='binary')
    print "Naive Bayes sklearn - Manual IDF: " + str(precision_recall_fscore_support(test_labels, classifier.predict(test_features), average='binary'))

    return generateJson("Naive Bayes (sklearn) - manual IDF", tr_acc, te_acc, f_score)


# k Nearest Neighbors - using manual IDF
def mClassifyKNN(training_set, test_set, n):
    print "K-Nearest Neighbor Classifier with {0} nearest neighbors. (m)".format(str(n))
    train_features, train_labels, test_features, test_labels = \
                                getFeaturesLabels(training_set, test_set)

    classifier = KNeighborsClassifier(n_neighbors=n)
    # print "Training..."
    classifier.fit(train_features, train_labels)
    # print "Training complete."
    tr_acc = accuracy_score(classifier.predict(train_features), train_labels)
    te_acc = accuracy_score(classifier.predict(test_features), test_labels)
    f_score = f1_score(test_labels, classifier.predict(test_features), average='binary')

    return generateJson("k-NN, k = %d - manual IDF" % n, tr_acc, te_acc, f_score)


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
    return float(j / len(labels))


def generateJson(algorithm, train_acc, test_acc, f_score):
    return {
        'Algorithm': algorithm,
        'Training Set accuracy' : "%.2f%%" % (train_acc * 100),
        'Test Set accuracy' : "%.2f%%" % (test_acc * 100),
        'F - Score' : "%.2f%%" % (f_score * 100)
    }

if __name__ == "__main__":
    main()
