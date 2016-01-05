import generalClassifier
import nltk

def main():
    generalClassifier.trainGeneralClassifier()
    tweet = raw_input("Enter tweet to classify: ")
    print generalClassifier.predict(tweet)

    return

if __name__ == "__main__":
    main()
