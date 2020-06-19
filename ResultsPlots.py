
# coding: utf-8

# In[32]:


import pymongo
from bson import ObjectId
import numpy as np
import pandas as pd
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator
from nltk import bigrams
import matplotlib.pyplot as plt
import itertools
import collections
import networkx as nx
from datetime import datetime

connection = pymongo.MongoClient('mongodb://localhost')
db = connection['posts']
collection = db['tweets']

cursor = collection.find({})

usernames = []
date = []
negative_dates = []
positive_dates = []
hashtagsDF = []
wordcloudsentim = []
wordcloudsentim_positive = []
wordcloudsentim_negative = []
affection_unsupervisedDF = []
affection_supervisedDF = []
sentiment_unsupervisedDF = []
sentiment_supervisedDF = []
countries = []

sadness_positive, fear_positive, surprise_positive, anger_positive, disgust_positive, joy_positive = 0,0,0,0,0,0
sadness_negative, fear_negative, surprise_negative, anger_negative, disgust_negative, joy_negative = 0,0,0,0,0,0
sadness_neutral, fear_neutral, surprise_neutral, anger_neutral, disgust_neutral, joy_neutral = 0,0,0,0,0,0

us_positive, uk_positive, nig_positive, sa_positive, ind_positive, can_positive, other_positive = 0,0,0,0,0,0,0
us_negative, uk_negative, nig_negative, sa_negative, ind_negative, can_negative, other_negative = 0,0,0,0,0,0,0

for x in cursor:
    usernames.append(x['username'])
    date.append(x['created'])
    hashtagsDF.append(x['preprocessed_text'][3])
    wordcloudsentim.append(x['preprocessed_text'][1])
    if(x['sentiment_analysis_unsupervised'] == 'positive'):
        wordcloudsentim_positive.append(x['preprocessed_text'][1])
    else:
        wordcloudsentim_negative.append(x['preprocessed_text'][1])
        
    affection_unsupervisedDF.append(x['affection_analysis_unsupervised'])
    sentiment_unsupervisedDF.append(x['sentiment_analysis_unsupervised'])
    sentiment_supervisedDF.append(x['sentiment_analysis_supervised'])
    
    if 'positive' in x['sentiment_analysis_supervised']:
        positive_dates.append(x['created'])
    elif 'negative' in x['sentiment_analysis_supervised']:
        negative_dates.append(x['created'])
    
#     if  x['sentiment_analysis_unsupervised'] == 'positive':
#         positive_dates.append(x['created'])
#     elif x['sentiment_analysis_unsupervised'] == 'negative':
#         negative_dates.append(x['created'])
    
    
    if 'affection_analysis_supervised' in x and x['affection_analysis_supervised'] != None:
        affection_supervisedDF.append(x['affection_analysis_supervised'])
    
    if 'country' in x and x['country'] != None:
        countries.append(x['country'])
        if x['country'] == 'United States of America' or x['country'] == 'United States':
            if 'positive' in x['sentiment_analysis_supervised']:
                us_positive+=1
            elif 'negative' in x['sentiment_analysis_supervised']:
                us_negative+=1
        elif x['country'] == 'United Kingdom':
            if 'positive' in x['sentiment_analysis_supervised']:
                uk_positive+=1
            elif 'negative' in x['sentiment_analysis_supervised']:
                uk_negative+=1
        elif x['country'] == 'Nigeria':
            if 'positive' in x['sentiment_analysis_supervised']:
                nig_positive+=1
            elif 'negative' in x['sentiment_analysis_supervised']:
                nig_negative+=1
        elif x['country'] == 'South Africa':
            if 'positive' in x['sentiment_analysis_supervised']:
                sa_positive+=1
            elif 'negative' in x['sentiment_analysis_supervised']:
                sa_negative+=1
        elif x['country'] == 'India':
            if 'positive' in x['sentiment_analysis_supervised']:
                ind_positive+=1
            elif 'negative' in x['sentiment_analysis_supervised']:
                ind_negative+=1
        elif x['country'] == 'Canada':
            if 'positive' in x['sentiment_analysis_supervised']:
                can_positive+=1
            elif 'negative' in x['sentiment_analysis_supervised']:
                can_negative+=1
        else:
            if 'positive' in x['sentiment_analysis_supervised']:
                other_positive+=1
            elif 'negative' in x['sentiment_analysis_supervised']:
                other_negative+=1
    
    
    if 'sadness' in x['affection_analysis_unsupervised']:
        if x['sentiment_analysis_unsupervised'] == 'positive':
            sadness_positive+=1
        elif x['sentiment_analysis_unsupervised'] == 'negative':
            sadness_negative+=1
        else:
            sadness_neutral+=1
    
    elif 'fear' in x['affection_analysis_unsupervised']:
        if x['sentiment_analysis_unsupervised'] == 'positive':
            fear_positive+=1
        elif x['sentiment_analysis_unsupervised'] == 'negative':
            fear_negative+=1
        else:
            fear_neutral+=1
            
    elif  'surprise' in x['affection_analysis_unsupervised']:
        if x['sentiment_analysis_unsupervised'] == 'positive':
            surprise_positive+=1
        elif x['sentiment_analysis_unsupervised'] == 'negative':
            surprise_negative+=1
        else:
            surprise_neutral+=1
        
    elif 'anger' in  x['affection_analysis_unsupervised']:
        if x['sentiment_analysis_unsupervised'] == 'positive':
            anger_positive+=1
        elif x['sentiment_analysis_unsupervised'] == 'negative':
            anger_negative+=1
        else:
            anger_neutral+=1
            
    elif 'disgust' in x['affection_analysis_unsupervised']:
        if x['sentiment_analysis_unsupervised'] == 'positive':
            disgust_positive+=1
        elif x['sentiment_analysis_unsupervised'] == 'negative':
            disgust_negative+=1
        else:
            disgust_neutral+=1
            
    elif 'joy' in  x['affection_analysis_unsupervised']:
        if x['sentiment_analysis_unsupervised'] == 'positive':
            joy_positive+=1
        elif x['sentiment_analysis_unsupervised'] == 'negative':
            joy_negative+=1
        else:
            joy_neutral+=1
    

    
 
#counts
counts_positive = [sadness_positive, fear_positive, surprise_positive, anger_positive, disgust_positive, joy_positive]
counts_negative = [sadness_negative, fear_negative, surprise_negative, anger_negative, disgust_negative, joy_negative]
counts_neutral = [sadness_neutral, fear_neutral, surprise_neutral, anger_neutral, disgust_neutral, joy_neutral]

countries_positive = [us_positive, uk_positive, nig_positive, sa_positive, ind_positive, can_positive, other_positive]
countries_negative = [us_negative, uk_negative, nig_negative, sa_negative, ind_negative, can_negative, other_negative]

print("Distinct tweets: ", len(usernames))
print("Distinct users: ", len(Counter(usernames).keys()))

stop_words = ['order','another','today','people','day','new', 'voice', 'earn', 'get', 'one','may','fan','let','showcase'
              ,'watch','home','time','join']

#Wordclouds
hashtagsDF2 = []
for listi in hashtagsDF:
    for i in listi:
        hashtagsDF2.append(i)

wordcloud = WordCloud(collocations=False).generate(' '.join(hashtagsDF2))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


wordcloudsentim2 = []
for listi in wordcloudsentim:
    for i in listi:
        wordcloudsentim2.append(i)

wordcloud1 = WordCloud(stopwords = stop_words, collocations=False).generate(' '.join(wordcloudsentim2))
plt.imshow(wordcloud1)
plt.axis("off")
plt.show()

wordcloudsentim3 = []
for listi in wordcloudsentim_positive:
    for i in listi:
        wordcloudsentim3.append(i)

wordcloud2 = WordCloud(stopwords = stop_words, collocations=False).generate(' '.join(wordcloudsentim3))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()


wordcloudsentim4 = []
for listi in wordcloudsentim_negative:
    for i in listi:
        wordcloudsentim4.append(i)

wordcloud3 = WordCloud(stopwords = stop_words, collocations=False).generate(' '.join(wordcloudsentim4))
plt.imshow(wordcloud3)
plt.axis("off")
plt.show()

#Timeseries analysis
only_date = []
only_date_positive = []
only_date_negative = []

for d in date:
    only_date.append(d.date())
for d in positive_dates:
    only_date_positive.append(d.date())
for d in negative_dates:
    only_date_negative.append(d.date())
    
date_keys = Counter(only_date).keys()
date_values = Counter(only_date).values()
date_keys, date_values = zip(*sorted(zip(date_keys, date_values)))
date_keys = date_keys[:-1]
date_values = date_values[:-1]

date_keys2 = Counter(only_date_positive).keys()
date_values2 = Counter(only_date_positive).values()
date_keys2, date_values2 = zip(*sorted(zip(date_keys2, date_values2)))
date_keys2 = date_keys2[:-1]
date_values2 = date_values2[:-1]

date_keys3 = Counter(only_date_negative).keys()
date_values3 = Counter(only_date_negative).values()
date_keys3, date_values3 = zip(*sorted(zip(date_keys3, date_values2)))
date_keys3 = date_keys3[:-1]
date_values3 = date_values3[:-1]

plot_df(date_keys, date_values, title="Tweets timeseries", xlabel='Date', ylabel='Number of tweets', dpi=100)
plot_df(date_keys2, date_values2, title="Positive tweets timeseries", xlabel='Date', ylabel='Number of positive tweets', dpi=100)
plot_df(date_keys3, date_values3, title="Negative tweets timeseries", xlabel='Date', ylabel='Number of negative tweets', dpi=100)

#Countries pie
countries = ['United States of America' if x=='United States' else x for x in countries]
countries_keys = Counter(countries).keys()
countries_values = Counter(countries).values()
countries_values = [round((x / len(countries))*100,3) for x in countries_values]
countries_values, countries_keys = zip(*sorted(zip(countries_values, countries_keys), reverse = True))
first_countries_keys = countries_keys[0:6]
first_countries_values = countries_values[0:6]
less_countries_values = countries_values[6:]
final_countries_keys = (*first_countries_keys, 'Other')
final_countries_values = first_countries_values + (round(sum(less_countries_values),3),)

explode = (0, 0.1, 0, 0, 0, 0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(final_countries_values, explode=explode, labels=final_countries_keys , autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

sentiment_for_countries(countries_positive,countries_negative)

#Affection bars
affection_unsupervisedDF2 = []
for listi in affection_unsupervisedDF:
    for i in listi:
        if i != 'None':
            affection_unsupervisedDF2.append(i)
            
aff_uns_keys = Counter(affection_unsupervisedDF2).keys()
aff_uns_values = Counter(affection_unsupervisedDF2).values()
aff_uns_values = [round((x / len(affection_unsupervisedDF2))*100,2) for x in aff_uns_values]
plt.bar(aff_uns_keys, aff_uns_values)
f1 = plt.figure()


affection_supervisedDF2 = []
for listi in affection_supervisedDF:
    for i in listi:
        if i != 'None':
            affection_supervisedDF2.append(i)
            
aff_s_keys = Counter(affection_supervisedDF2).keys()
aff_s_values = Counter(affection_supervisedDF2).values()
aff_s_values = [round((x / len(affection_supervisedDF2))*100,2) for x in aff_s_values]


f2= plt.figure(figsize=(7,4)) 
ax = plt.subplot(111)
width=1
ax.bar(range(0,len(aff_s_keys)), aff_s_values, width=width/2)
ax.set_xticks(np.arange(0,len(aff_s_values)) + width/2)
ax.set_xticklabels(aff_s_keys)

locs, labels = plt.xticks() #gets labels
plt.setp(labels, rotation=90) #sets rotation of the labels
plt.show()


#Sentiment bars
sen_uns_keys = Counter(sentiment_unsupervisedDF).keys()
sen_uns_values = Counter(sentiment_unsupervisedDF).values()
sen_uns_values = [round((x / len(sentiment_unsupervisedDF))*100,2) for x in sen_uns_values]
plt.bar(sen_uns_keys, sen_uns_values)
f3 = plt.figure()


sentiment_supervisedDF2 = []
for listi in sentiment_supervisedDF:
    for i in listi:
        if i != 'None':
            sentiment_supervisedDF2.append(i)

sen_s_keys = Counter(sentiment_supervisedDF2).keys()
sen_s_values = Counter(sentiment_supervisedDF2).values()
sen_s_values = [round((x / len(sentiment_supervisedDF2))*100,2) for x in sen_s_values]
plt.bar(sen_s_keys, sen_s_values)
f4 = plt.figure()


ax1 = f1.add_subplot(111)
ax1.plot()

sentiment_for_affections(counts_positive,counts_negative,counts_neutral)

hashtag_network(hashtagsDF)


# In[31]:


def plot_df(keys, values, title, xlabel, ylabel, dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(keys, values, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()


# In[30]:


def sentiment_for_affections(counts_positive,counts_negative,counts_neutral):
    N = 6

    ind = np.arange(N) # the x locations for the groups
    width = 0.35
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(ind, counts_positive, width, color='g')
    ax.bar(ind, counts_negative, width, color='r')
    ax.bar(ind, counts_neutral, width, color='b')

    xl = ['sadness','fear','surprise','anger','disgust','joy'] 

    ax.set_ylabel('Affections')
    ax.set_title('Positive and negative affections segments')
    ax.set_xticklabels(xl)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 25000, 5000))
    ax.legend(labels=['Positive','Negative','Neutral'])
 
    plt.show()


# In[29]:


def sentiment_for_countries(countries_positive,countries_negative):
    N = 7

    ind = np.arange(N) # the x locations for the groups
    width = 0.35
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(ind, countries_negative, width, color='r')
    ax.bar(ind, countries_positive, width, color='g')

    xl = ['USA', 'UK', 'Nigeria', 'S. Africa', 'India', 'Canada', 'Other'] 

    ax.set_ylabel('Countries')
    ax.set_title('Positive and negative countries segments')
    ax.set_xticklabels(xl)
    ax.set_xticks(np.arange(0, 7, 1))
    #ax.set_yticks(np.arange(0, 20000, 5000))
    ax.legend(labels=['Positive','Negative'])
    #fig.savefig('countriessen.png')

    plt.show()


# In[33]:


def hashtag_network(hashtagsDF):
    from nltk import bigrams
    terms_bigram = [list(bigrams(hashtag)) for hashtag in hashtagsDF]
    bigrams = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigrams)
    
    bigram_df = pd.DataFrame(bigram_counts.most_common(30), columns=['bigram', 'count'])
       
    d = bigram_df.set_index('bigram').T.to_dict('records')
    # Create network plot 
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, k=2)

    # Plot networks
    nx.draw_networkx(G, pos, font_size=16, width=3, edge_color='grey', node_color='purple', with_labels = False, ax=ax)

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.135, value[1]+.045
        ax.text(x, y, s=key, bbox=dict(facecolor='red', alpha=0.25),horizontalalignment='center', fontsize=8)
 
    plt.show()

