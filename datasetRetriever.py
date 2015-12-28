from tweetTemplate import *
import pickle
import random

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

def prepareDatasets():
    # Get datasets
    sarcasm_set = getLabelledDataset("../data/sarcasm_classified_set.tds")
    london_set = getLabelledDataset("../data/london_classified_set.tds")
    # Shuffle twice
    combined_set = random.shuffle(sarcasm_set + london_set)
    combined_set = random.shuffle(sarcasm_set + london_set)
    # Write to files
    trainFile = open("../data/training_set.tds", 'w')
    splitPoint = int(len(combined_set)*0.9)
    for tweet, label in combined_set[:splitPoint]:
        pickle.dump(LabelledTemplate(tweet, label))
    trainFile.close()
    testFile = open("../data/test_set.tds", 'w')
    for tweet, label in combined_set[splitPoint:]:
        pickle.dump(LabelledTemplate(tweet, label))
    testFile.close()

def getTrainingDataset():
    return getLabelledDataset("../data/training_set.tds")

def getTestDataset():
    return getLabelledDataset("../data/test_set.tds")
