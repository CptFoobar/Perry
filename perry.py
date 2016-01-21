import generalClassifier
import nltk

def main():
    for i in range(10):
        clf = generalClassifier.trainGeneralClassifier()
    #tweet = raw_input("Enter tweet to classify: ")
    #print generalClassifier.predict(tweet, clf)
    return

if __name__ == "__main__":
    main()
