from tweepy.streaming import StreamListener
from tweepy import Stream
import twitterAuth
import pickle
from tweetTemplate import tweetTemplate

class TweetListener(StreamListener):

    def __init__(self):
        super(TweetListener, self).__init__()
        self.tweetFile = open("../data/london_tweets_dataset.tds", 'a')
        self.continueListening = True
        self.maxTweets = 10000
        self.currentCount = 0

    def on_status(self, status):
        tweetObj = tweetTemplate(status.text)
        pickle.dump(tweetObj, self.tweetFile)
        self.currentCount += 1
        print "--------------------------------"
        print "london[" + str(self.currentCount) + "]: " + status.text
        return self.continueListening and (self.currentCount < self.maxTweets)

    def on_error(self, status):
        print "london: " + status
        return False

    def stopListening(self):
        self.continueListening = False
        self.tweetFile.close()

def main():
    auth = twitterAuth.getAuthentication()
    tl_l = TweetListener()
    london_stream = Stream(auth, tl_l)
    print "Listening for London tweets..."
    ## Co-ords
    london = [-0.4958, 51.2825, 0.2211, 51.6828]
    try:
        london_stream.filter(locations=london)
    except KeyboardInterrupt:
        tl_l.stopListening()
        print "\nListening stopped."
    print "\nCollected 10,000 tweets from London"


if  __name__ == '__main__':
    main()
