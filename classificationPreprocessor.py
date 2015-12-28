from tweetTemplate import *
from HTMLParser import HTMLParser
import datasetRetriever
import langid
import re
import string
import pickle
import sys

class HTMLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def stripHtmlTags(html):
    s = HTMLStripper()
    s.feed(html)
    return s.get_data()


def preprocess(topic):
    print "Processing tweets, topic: " + topic
    tweetSet = datasetRetriever.getTweetDataset("../data/" + topic + "_tweets_dataset.tds")
    outFile = open("../data/" + topic + "_processed_dataset.tds", 'w')
    count = 0
    for tweet in tweetSet:
        tweet = ' '.join(tweet.split())
        lang, conf = langid.classify(tweet)
        count += 1
        if count % 1000 == 0:
            print "At {0}th tweet.".format(str(count))
        if lang == 'en' and conf > 0.99:
            goodTweet, tweet = processTweet(tweet)
            if goodTweet:
                pickle.dump(tweetTemplate(tweet), outFile)
        else:
            continue
    outFile.close()
    totalCount = len(tweetSet)
    tweetSet = datasetRetriever.getTweetDataset("../data/" + topic + "_processed_dataset.tds")
    print "Processing completete. \nTotal:" + str(totalCount) + " tweets. Usable: " + str(len(tweetSet))
    return


def processTweet(tweet):
    goodTweet = False

    # Drop tweets from Yes!News.. they use #noticas hashtags which adds noise to dataset
    if tweet.find("Yes!News") > -1:  return False, None
    # Drop retweets. We might already have them
    if tweet.find("RT") == 0: return False, None

    # Replace URLs
    tweet = re.sub("((https?|ftp)://|(www|ftp)\\.)?[a-z0-9-]+(\\.[a-z0-9-]+)+([/?].*)?$", "URL", tweet)

    # Remove Non-ASCII
    tweet = ''.join([i if ord(i) < 128 else '' for i in tweet])

    # Remove digits
    # Forget them
    #for digit in string.digits: tweet = tweet.replace(digit, '')

    # Remove Unicode emojis
    try:
    # UCS-4
        emojiRe = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
    # UCS-2
        emojiRe = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

    emojiRe.sub('', tweet)

    # Replace @refs
    tweet = re.sub(r"@\w+", "REF", tweet)

    # Remove HTML tags
    tweet = stripHtmlTags(tweet)

    if len(tweet.split()) < 4 or len(minimizedTweet(tweet).split()) < 3: goodTweet = False
    else: goodTweet = True

    return (goodTweet, tweetTemplate(' '.join(tweet.split())))

# Return tweet after stripping URLs, REFs and TAGs
def minimizedTweet(tweet):
    tweet = re.sub(r"URL", '', tweet)
    tweet = re.sub(r"TAG", '', tweet)
    return re.sub(r"REF", '', tweet)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        printUsage()
    else:
        if sys.argv[1] == '--london':
            preprocess("london")
        elif sys.argv[1] == '--sarcasm':
            preprocess("sarcasm")
        elif sys.argv[1] == '--sf':
            preprocess("sf")
        else: printUsage()

def printUsage():
    print "Usage: python classificationPreprocess.py --<topic>\n" + \
            "Topics:\n--london: Process tweets from London\n" + \
            "--sarcasm: Process tweets containing Sarcasm indicator tags\n" + \
            "--sf: Process Global tweet corpus"


if __name__ == "__main__":
    main()
