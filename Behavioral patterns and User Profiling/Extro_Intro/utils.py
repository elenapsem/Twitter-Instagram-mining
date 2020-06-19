import csv
import process_csv
import pandas as pd
import re

dataset = pd.read_csv("NRC-Emotion.csv", index_col=0, delimiter=';')
print(dataset)
df = dataset.drop('emotion-intensity-score', axis=1)
print(df)
words_lexicon = df.index.tolist()


tweets = pd.read_csv("processing_extro_intro_tweets.csv", index_col=0)
tweets = tweets[tweets['word'].notna()]
sentiment_words = tweets.loc[:, "word"].tolist()
print(sentiment_words)
list_sentiment_words = []
for x in sentiment_words:
    i = 0
    y = x.split(',')
    for z in y:
        if z.strip() in words_lexicon:
            i = i+1
    list_sentiment_words.append(i)
print(list_sentiment_words)

tweets['len_emotion'] = list_sentiment_words
print(tweets)
tweets.to_csv("with_len_emotions.csv", index=False)






