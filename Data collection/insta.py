import pymongo
from instaloader import Instaloader, Profile
from langdetect import DetectorFactory, detect

connection = pymongo.MongoClient('mongodb://localhost')
db = connection['tweets']
collection = db['collection']

loader = Instaloader()
NUM_POSTS = 10

loader.login('iro_tsantalidou', 'Just_Do_It!1996')
DetectorFactory.seed = 0

def get_hashtags_posts(query):
    posts = loader.get_hashtag_posts(query)
    for post in posts:
        if detect(post.caption) == 'en':
            followers = post.owner_profile.followers
            username = post.owner_profile.username
            hashtags = post.caption_hashtags
            text = post.caption
            likes = post.likes
            cor = None
            location = None
            if post.location != None:
                cor = [post.location.lat, post.location.lng]
                location = post.location.name
            created = post.date

            tweet = {'username': username, 'followers': followers, 'text': text, 'likes': likes,
                     'location': location, 'coordinates': cor, 'hashtags': hashtags, 'created': created}

            #Save the refined Tweet data to MongoDB
            collection.insert_one(tweet)

if __name__ == "__main__":
    hashtag = "stayAtHome"
    get_hashtags_posts(hashtag)

