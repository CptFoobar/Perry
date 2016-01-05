import datasetRetriever
import langid
import pickle
from tweetTemplate import *

def main():

    tweetSet = datasetRetriever.getTweetDataset("../data/london_processed_dataset.tds")
    outFile = open("../data/london_classified_set.tds", 'a')
    count = 0
    total = 0
    previousCount = 225 + 257 + 386 + 451 + 751 # sarcasm: 225 + 257 + 386 + 301
    for i in range(previousCount-1, len(tweetSet)):
        tweet = tweetSet[i]
        total += 1
        if total % 10 == 0: print "{0}th tweet".format(str(total))
        tweet = tweet.tweet
        tweet = ' '.join(tweet.split())
        lang, conf = langid.classify(tweet)
        if lang == 'en' and conf > 0.99:
            print "Tweet[{0}]: {1}".format(str(total), tweet)
            rating = raw_input("Rate: {0, 1, 2, 3, 4}, D: Discard, Q: Quit - ")
            if rating in "01234":
                pickle.dump(LabelledTemplate(tweet, rating), outFile)
                count += 1
            elif rating.lower() == 'q':
                break
    outFile.close()
    print "Classification complete.\n Total classified: {0}, {1} rated, {2} discarded"\
                    .format(str(total), str(count), str(total - count))
    return

if __name__ == "__main__":
    main()
