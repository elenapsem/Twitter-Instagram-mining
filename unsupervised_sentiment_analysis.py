from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from py_lex import EmoLex

lexicon = EmoLex('./emo-lex.txt')


# function to return sentiments based on Vader Sentiment Analyzer
def sentiment_score(sentence):

    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        return 'positive'

    elif sentiment_dict['compound'] <= - 0.05:
        return 'negative'

    else:
        return 'neutral'

# function to return emotions of sentences based on NRC lexicon
def emotion(text):
    annotation = lexicon.annotate_doc(text)
    for x in annotation:
        if 'positive' in x:
            x.remove('positive')
        if 'negative' in x:
            x.remove('negative')
        if 'anticipation' in x:
            x.remove('anticipation')
        if 'trust' in x:
            x.remove('trust')
    summary = lexicon.summarize_annotation(annotation, text)
    max_value = max(summary.values())  # maximum value
    if max_value != 0.0:
        max_keys = [k for k, v in summary.items() if v == max_value]  # getting all keys containing the `maximum`
    else:
        max_keys = ['None']
    return max_keys
