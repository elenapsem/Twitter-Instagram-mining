import pickle

tfidf = pickle.load(open("vectorizer.pickle", "rb"))
anger = pickle.load(open("anger.sav", "rb"))
anticipation = pickle.load(open("anticipation.sav", "rb"))
fear = pickle.load(open("fear.sav", "rb"))
disgust = pickle.load(open("disgust.sav", "rb"))
sadness = pickle.load(open("sadness.sav", "rb"))
surprise = pickle.load(open("surprise.sav", "rb"))
joy = pickle.load(open("joy.sav", "rb"))
trust = pickle.load(open("trust.sav", "rb"))
sentiment = pickle.load(open("sentiment_classifier.sav", "rb"))


def emotion_classification(tweet):
    emotion = []
    tweet = [' '.join(tweet)]
    tweet = tfidf.transform(tweet)
    x = int(anger.predict(tweet))
    if x ==1:
        emotion.append('anger')
    x = int(anticipation.predict(tweet))
    if x == 1:
        emotion.append('anticipation')
    x = int(fear.predict(tweet))
    if x == 1:
        emotion.append('fear')
    x = int(disgust.predict(tweet))
    if x == 1:
        emotion.append('disgust')
    x = int(sadness.predict(tweet))
    if x == 1:
        emotion.append('sadness')
    x = int(surprise.predict(tweet))
    if x == 1:
        emotion.append('surprise')
    x = int(joy.predict(tweet))
    if x == 1:
        emotion.append('joy')
    x = int(trust.predict(tweet))
    if x == 1:
        emotion.append('trust')
    return emotion

def sentiment_classification(tweet):
    sentiments = []
    tweet = [' '.join(tweet)]
    tweet = tfidf.transform(tweet)
    x = int(sentiment.predict(tweet))
    if x ==1:
        sentiments.append('positive')
    else:
        sentiments.append('negative')

    return sentiments