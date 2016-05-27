from tweetTemplate import *
import pickle
import random
import re
import string
import itertools
import sys
from math import log
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from nltk.corpus import words

# TF-IDF Vectorizer
vectorizer = None
testSet = []
trainSet = []

def init_all(mf):
    init(mf)
    getTrainingSet()

def init(mf):
    global vectorizer
    parameters = {
        'ngram_range': (1, 3),
        'stop_words': 'english',
        'lowercase': True,
        'strip_accents': 'unicode',
        'max_features': mf,
        'use_idf': False,
        'smooth_idf': False
    }
    vectorizer = TfidfVectorizer(**parameters)

def getTweetDataset(fileName):
    filein = open(fileName, 'r')
    tweetSet = []
    while True:
        try:
            tweetObj = pickle.load(filein)
            tweetSet.append(tweetObj.tweet)
        except EOFError:
            break
        except Exception:
            # print "Exception"
            # silently ignore exceptions
            continue
    filein.close()
    return tweetSet

def getLabelledDataset(fileName):
    filein = open(fileName, 'r')
    labelledSet = []
    while True:
        try:
            tweetObj = pickle.load(filein)
            labelledSet.append((tweetObj.tweet, tweetObj.label))
        except EOFError:
            break
        except AttributeError:
            print "The file does not contain labelled tweets. Please check the path"
            break
        except Exception:
            # print "Exception"
            # silently ignore exceptions
            continue
    filein.close()
    return labelledSet

def prepareDatasets(splitFactor):
    # Get datasets
    print "Preparing datasets"
    sarcasm_set = getLabelledDataset("../data/sarcasm_classified_set.tds")
    london_set = getLabelledDataset("../data/london_classified_set.tds")
    combined_set = []
    for tweet, label in sarcasm_set + london_set:
        if len(label) == 1:
            label = 1 if int(label) > 2 else 0
            combined_set.append((tweet, label))

    # Shuffle twice
    random.shuffle(combined_set)
    random.shuffle(combined_set)

    # Write to files
    splitPoint = int(len(combined_set)*splitFactor)
    trainFile = open("../data/training_set.tds", 'w')
    for tweet, label in combined_set[:splitPoint]:
        pickle.dump(LabelledTemplate(tweet, label), trainFile)
    trainFile.close()
    testFile = open("../data/test_set.tds", 'w')
    for tweet, label in combined_set[splitPoint:]:
        pickle.dump(LabelledTemplate(tweet, label), testFile)
    testFile.close()

def getTrainingDataset():
    labelledBigrams = getLabelledBigrams("../data/training_set.tds")
    allBigrams = []
    allBigrams.extend(bigram for bigram in (bigrams for bigrams, label in labelledBigrams))
    allBigrams = list(itertools.chain(*allBigrams))
    bigramFeaturesWgt = getWeightedFeatures(allBigrams, labelledBigrams)
    bigramFeaturesWgt = sorted(set(bigramFeaturesWgt), key = lambda feature: feature[1], reverse = True)
    threshWgt = (min([weight for bigram, weight in bigramFeaturesWgt]) + \
             max([weight for bigram, weight in bigramFeaturesWgt]))
    threshWgt = max([weight for bigram, weight in bigramFeaturesWgt])
    bigramFeatures = [bigram for bigram, weight in bigramFeaturesWgt if weight >= threshWgt]
    return bigramFeatures, labelledBigrams

def getTestDataset():
    return getLabelledBigrams("../data/test_set.tds")

def getLabelledBigrams(filepath):
    tweetSet = getLabelledDataset(filepath)
    tweetSet = cleanSet(tweetSet)
    tokenized = [(nltk.word_tokenize(tweet), label) for tweet, label in tweetSet]
    filteredTokenized = filterStopWords(tokenized)
    labelledBigrams = []
    for tokens, label in filteredTokenized:
        bigrams = nltk.bigrams(tokens)
        labelledBigrams.append(([bigram for bigram in bigrams], label))
    return labelledBigrams

def cleanSet(tweets):
    cleanedSet = []
    wrds = words.words()
    count = 0
    for tweet, label in tweets:
        count += 1
        cleanedSet.append((cleanTweet(tweet, wrds), label))
        sys.stdout.write("\rCleaning tweets: %d%%" % int(count * 100 / len(tweets)))
        sys.stdout.flush()
    print
    return cleanedSet

def cleanTweet(tweet, wrds):
    # Remove non important hashtags
    SPECIAL_TAGS = [
        "#notreally",
        "#not",
        "#sarcasm"
    ]
    tweet = re.sub(r"#\w+", lambda match: \
            "S_TAG" if match.group(0) in SPECIAL_TAGS else "", tweet)

    # Remove Punctuations
    for punc in string.punctuation:
        tweet = tweet.replace(punc, '')

    # Remove digits
    tweet = re.sub(r"\d+", '', tweet)

    # Remove words not in english dictionary (nltk.words corpus approximates eng dict)
    tweet = ' '.join([w for w in tweet.split() if w in wrds])

    return tweet.lower()

def getWeightedFeatures(allGrams, labelledGrams):
    # IDF weighting
    N = len(labelledGrams)
    weightedFeatures = []
    for gram in allGrams:
        count = 0
        for grams, label in labelledGrams:
            if gram in grams:
                count += 1
        weightedFeatures.append((gram, log(N / count)))
    return weightedFeatures

def filterStopWords(tokenized):
    filtered = []
    for tokens, label in tokenized:
        filteredTokens = [token for token in tokens if token not in stopwords.words('english')]
        filtered.append((filteredTokens, label))
    return filtered


def getBigrams(tweet):
    tweet = cleanTweet(tweet, words.words())
    tokens = nltk.word_tokenize(tweet)
    filteredTokens = [token for token in tokens if token not in stopwords.words('english')]
    tweetBigrams = [bigram for bigram in nltk.bigrams(filteredTokens)]
    return tweetBigrams


def getTrainingSet():
    global trainSet
    if len(trainSet) == 0:
        ts = getLabelledDataset("../data/training_set.tds")
        ts = cleanSet(ts)
        trainSet = list(ts)
    else:
        ts = trainSet
    tweetSet, labels = [list(item) for item in zip(*ts)]
    return vectorizer.fit_transform(tweetSet).todense(), labels

def getTestSet():
    global testSet
    if len(testSet) == 0:
        ts = getLabelledDataset("../data/test_set.tds")
        ts = cleanSet(ts)
        testSet = list(ts)
    else:
        ts = testSet
    tweetSet, labels = [list(item) for item in zip(*ts)]
    return vectorizer.transform(tweetSet), labels


def getVectorizer():
    return vectorizer

def vectorizeOne(tweet):
    return vectorizer.transform([tweet])[0]
