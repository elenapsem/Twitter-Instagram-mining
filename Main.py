import pickle
import re
from datetime import datetime

import gensim
import pandas
import pymongo
from bson import ObjectId
import Preprocessing
import geo_info
#import like_prediction
import like_prediction
import unsupervised_sentiment_analysis
import Topic_analysis
import supervised_analysis

connection = pymongo.MongoClient('mongodb://localhost')
db = connection['posts']
collection = db['tweets']
# collection2 = db['processed']


# x = '5ea72e14968403822e5b79d5'
# cursor = collection.find({'_id': ObjectId(x)})
# for k in cursor:
#     print(k)


# ============= preprocessing step =============== #
# cursor = collection.find({})
# for x in cursor:
#     collection.update_one(
#         {"_id": x['_id']},
#         {
#             "$set": {
#                 "preprocessed_text": Preprocessing.preprocessing(x['text'])
#             }
#         }
#     )
#
# cursor = collection.find({})
# result =[]
# for x in cursor:
#      r = re.search("giveaway", x['text'].lower())
#      if r != None or len(x['preprocessed_text'][1])<3:
#         collection.delete_one({'_id':x['_id']})


# ============= unsupervised sentiment and emotion classification ===============#
# cursor = collection.find({})
# for x in cursor:
#     collection.update_one(
#         {"_id": x['_id']},
#         {
#             "$set": {
#                 "affection_analysis_unsupervised": unsupervised_sentiment_analysis.emotion(x['preprocessed_text'][1] + x['preprocessed_text'][2])
#             }
#         }
#     )
#

# cursor = collection.find({})
# for x in cursor:
#     collection.update_one(
#         {"_id": x['_id']},
#         {
#             "$set": {
#                 "sentiment_analysis_unsupervised": unsupervised_sentiment_analysis.sentiment_score(x['preprocessed_text'][0])
#             }
#         }
#     )

# ============ supervised sentiment and emotion classification ===================#

# cursor = collection.find({})
# for x in cursor:
#     collection.update_one(
#         {"_id": x['_id']},
#         {
#             "$set": {
#                 "affection_analysis_supervised": supervised_analysis.emotion_classification(x['preprocessed_text'][1] + x['preprocessed_text'][2])
#             }
#         }
#     )


# cursor = collection.find({})
# for x in cursor:
#     collection.update_one(
#         {"_id": x['_id']},
#         {
#             "$set": {
#                 "sentiment_analysis_supervised": supervised_analysis.sentiment_classification(x['preprocessed_text'][1] + x['preprocessed_text'][2])
#             }
#         }
#     )


# ====================== topic analysis ======================= #
result =[]
cursor = collection.find({})
id = []
for x in cursor:
    result.append(x['preprocessed_text'][1])
    id.append(x['_id'])

topics_list, topics = Topic_analysis.topic_analysis(result)
print(topics_list)
print(topics[1])

# with open('topics.txt', 'w') as f:
#     for item in topics_list:
#         print(item)
#         # print('\n')
#         f.write(str(item))
#         f.write("\n")

for i in range(len(id)):
    collection.update_one(
        {"_id": ObjectId(id[i])},
        {
            "$set": {
                "topic": topics[i]
            }
        }
    )


#=============== geo information analysis ======================== #

#52892
# cursor = collection.find({})
# for x in cursor:
#     if x['coordinates'] == None and x['location'] != None:
#         cor, country = geo_info.geocoding(x['location'])
#         collection.update_one(
#             {"_id": x['_id']},
#             {
#                 "$set": {
#                     "coordinates": cor
#                 }
#             }
#         )
#         collection.update_one(
#             {"_id": x['_id']},
#             {
#                 "$set": {
#                     "country": country
#                 }
#             }
#         )
#     elif x['coordinates'] == None and x['location'] == None:
#         collection.update_one(
#             {"_id": x['_id']},
#             {
#                 "$set": {
#                     "country": None
#                 }
#             }
#         )
#     elif x['coordinates'] != None:
#         collection.update_one(
#             {"_id": x['_id']},
#             {
#                 "$set": {
#                     "country": geo_info.find_country(x['coordinates'])
#                 }
#             }
#         )
#

# country = []
# cursor = collection.find({})
# for x in cursor:
#     if 'country' in x and x['country'] != None:
#         country.append(x['country'])
#
# geo_info.draw_map(country)

# ======================== like - prediction ========================================================#

# cursor = collection.find({})
# date = datetime(2020, 4, 27)
# tweets = []
# i = 0
# for x in cursor:
#     diff = date - x['created']
#     if diff.days > 1:
#         tweets.append(x)
#
# like_prediction.create_df(tweets)

#like_prediction.classification_without_text('like_without_text')
#like_prediction.classification_with_text('like_with_text')
#like_prediction.plotting("like_without_text.csv", "like_without_text")
#like_prediction.plotting("like_with_text.csv", "like_with_text")

