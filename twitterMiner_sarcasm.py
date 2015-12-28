from tweepy.streaming import StreamListener
from tweepy import Stream
import twitterAuth
import pickle
from tweetTemplate import tweetTemplate

class TweetListener(StreamListener):

    def __init__(self):
        super(TweetListener, self).__init__()
        self.tweetFile = open("../data/sarcasm_tweets_dataset.tds", 'a')
        self.continueListening = True

    def on_status(self, status):
        print "--------------------------------"
        print "sarcasm: " + status.text
        tweetObj = tweetTemplate(status.text)
        pickle.dump(tweetObj, self.tweetFile)
        return self.continueListening

    def on_error(self, status):
        print "sarcasm: " + status
        return False

    def stopListening(self):
        self.continueListening = False
        self.tweetFile.close()

def main():
    auth = twitterAuth.getAuthentication()
    tl_s = TweetListener()
    sarcasm_stream = Stream(auth, tl_s)
    print "Listening for #sarcasm tweets..."
    try:
        sarcasm_stream.filter(track=["#not", "#notreally", "#sarcasm"])
    except KeyboardInterrupt:
        tl_s.stopListening()
        print "\nListening stopped"

if  __name__ == '__main__':
    main()
