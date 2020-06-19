import pymongo
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import datetime
import json

connection = pymongo.MongoClient('mongodb://localhost')
db = connection['posts']
collection = db['tweets']

# Add the keywords you want to track. They can be cashtags, hashtags, or words.
WORDS = ['#stayathome', '#StayAtHome']

language = ['en']

consumer_key = 'zUzinrDvmkUuoCLUEvSoVgf6r'
consumer_secret = 'tmpBBKItPlV3oetcBOYsCM0vdf3C3dvyGNUkFRRL8hOpD4EMh9'
access_token = '3355948923-emPMmXurYeiEsLzL8sCzAueZgO8FLxUIgOCdySc'
access_token_secret = 'nwTnB5xwCtu39erqAW9NLvXZo93aLVPONRFtvJ94KxXRG'


class StreamListener(tweepy.StreamListener):
    #     access the twitter streaming api, class from tweepy
    def on_connect(self):
        # called initially to connect to the Streaming API
        print("You are now connected to the Streaming API.")

    def on_error(self, status_code):
        # on error - if an error occurs, display the error / status code
        print('An error has occured: ' + repr(status_code))
        return False

    def on_data(self, data):
        # it connects to your mongoDB and stores the tweet
        try:
            # Load the Tweet into the variable "t"
            t = json.loads(data)

            # Pull important data from the tweet to store in the database.
            #tweet_id = t['id_str']  # The Tweet ID from Twitter in string format
            username = t['user']['screen_name']  # The username of the Tweet author
            followers = t['user']['followers_count']  # The number of followers the Tweet author has
            location = t['user']['location']
            cor = t['coordinates']
            if "retweeted_status" in t:
                t = t["retweeted_status"]

                # In data we may have the pure data or as data we have the retweeted_status
                # Firstly as tweet we establish the simple text.
                # If there is extended_tweet in data, there is the full_text
            text = t["text"]
            if "extended_tweet" in t:
                text = t["extended_tweet"]["full_text"]
            likes = t['favorite_count']
            hashtags = t['entities']['hashtags'] # Any hashtags used in the Tweet
            dt = t['created_at']  # The timestamp of when the Tweet was created

            # Convert the timestamp string given by Twitter to a date object called "created". This is more easily manipulated in MongoDB.
            created = datetime.datetime.strptime(dt, '%a %b %d %H:%M:%S +0000 %Y')

            # Load all of the extracted Tweet data into the variable "tweet" that will be stored into the database
            tweet = {'username': username, 'followers': followers, 'text': text, 'likes': likes,
                     'location': location, 'coordinates': cor, 'hashtags': hashtags, 'created': created}


            # Save the refined Tweet data to MongoDB
            collection.insert_one(tweet)

            # Optional - Print the username and text of each Tweet to your console in realtime as they are pulled from the stream
            print(username + ':' + ' ' + text)
            return True
        except Exception as e:
            print(e)

        #     t = json.loads(data)
        #     # grab the created_at data from the Tweet to use for display and change it to Data object
        #     created_at = t['created_at']
        #     dt = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
        #     t['created_at'] = dt
        #     # print a message to the screen that we have collected a tweet
        #     print('tweet inserted')
        # except Exception as e:
        #     print(e)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# set up the listener. the wait_on_rate_limit=true is needed to help with twitter api rate limiting.True
listener = StreamListener(
    api=tweepy.API(wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True))
streamer = tweepy.Stream(auth=auth, listener=listener, tweet_mode='extended', until='2020-04-22')
print("Tracking: " + str(WORDS))
streamer.filter(track=WORDS, languages=language)