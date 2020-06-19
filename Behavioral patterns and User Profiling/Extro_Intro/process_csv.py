import pandas as pd
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pickle
import preprocessor as p
import re
from spellchecker import SpellChecker


porter_stemmer = PorterStemmer()
wordNetLemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# function to correct misspellings in words
def correct_spellings(text):
    spell = SpellChecker()
    corrected_text = []
    misspelled_words = spell.unknown(text)
    for word in text:
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return corrected_text

# function to tokenize and lemmatize sentences
def data_preprocessing(text):
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    word_tokens = word_tokenize(text)
    word_tokens = [w.lower() for w in word_tokens]
    word_tokens = [w for w in word_tokens if not w in stop_words]
    word_tokens = correct_spellings(word_tokens)
    pos_tagged_text = pos_tag(word_tokens)
    word_tokens = [wordNetLemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in
                   pos_tagged_text]
    # word_tokens = [porter_stemmer.stem(w) for w in word_tokens]
    word_tokens = [w for w in word_tokens if len(w) >= 3]
    return word_tokens

# function to remove punctuation
def remove_punctuation(text):
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[\d]', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# function to clean the tweets
def tweet_cleaning(text):
    parsed = p.parse(text)
    emojis = [x.match for x in parsed.emojis] if not parsed.emojis is None else []
    hashtags = [x.match for x in parsed.hashtags] if not parsed.hashtags is None else []
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.NUMBER)
    text = p.clean(text)
    p.set_options(p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.HASHTAG)
    text2 = p.clean(text)
    return [text, text2, emojis, hashtags]

# function to process emojis
def convert_emojis_to_word(emojis):
    text = " "
    text = text.join(emojis)
    with open('Emoji_Dict.p', 'rb') as fp:
        Emoji_Dict = pickle.load(fp)
    Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}
    for emot in Emoji_Dict:
        text = re.sub(r'(' + emot + ')', "_".join(Emoji_Dict[emot].replace(",", "").replace(":", "").split()), text)
    text = re.sub(r'_', " ", text)
    if text == '':
        return []
    else:
        return list(set(word_tokenize(text)))


def preprocessing(text):
    lis = tweet_cleaning(text)
    lis[2] = convert_emojis_to_word(lis[2])
    text_n = remove_punctuation(lis[1])
    lis[1] = data_preprocessing(text_n)
    return [lis[1], lis[2], lis[3]]