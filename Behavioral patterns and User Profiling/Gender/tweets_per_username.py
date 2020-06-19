# geolocation, gender, age, pages that he likes, followers, following
import pandas
import pymongo
import tweepy

client = pymongo.MongoClient('mongodb://localhost')
db = client['tweetsofusers']
collection_tweets = db['tweets_from_users']
collection_users = db['usernames']


consumer_key = 'THDW6zSjuRRhfBACtjSJZ7Jw7'
consumer_secret = 'hlbbvrQAoKhgXOiEvlK7yMVNfn6fLrvy5okdZ1aD6sWi8u487V'
access_key = '3355948923-3VgbhIj6Ljt7BLY9PPPrmkQGtwsH2Hud8o1J6l2'
access_secret = '5wz3DxnT9xE6u2Yb3zqQ9baaUO9SQO9Nu6i0c1mxS6l72'


def get_tweets(username):
    # set count to however many tweets you want
    number_of_tweets = 100

    # get tweets
    for tweet in tweepy.Cursor(api.user_timeline, screen_name=username, tweet_mode="extended").items(number_of_tweets):
        print(tweet)
        # create dictionary of tweet information: username, tweet id, date/time, text

        try:
            t = {'username': username, 'id': tweet.id, 'id_str': tweet.id_str,
                'retweet_text': tweet.retweeted_status.full_text, 'entities': tweet.entities,
                'date': tweet.created_at, 'coordinates': tweet.coordinates}
        except:
            t = {'username': username, 'id': tweet.id, 'id_str': tweet.id_str,
                 'text': tweet.full_text, 'entities': tweet.entities,
                 'date': tweet.created_at, 'coordinates': tweet.coordinates}
        print(t)
        collection_tweets.insert_one(t)


def get_user(username):
    api = tweepy.API(auth)
    user = api.get_user(screen_name=username)
    print(user)
    u = {'username': user.screen_name, 'name': user.name, 'id': user.id, 'location': user.location, 'description': user.description,
         'followers_count': user.followers_count, 'friends_count': user.friends_count,
         'favourites_count': user.favourites_count, 'statuses_count': user.statuses_count, 'verified': user.verified}
    # favourites_count is the number of tweets that given user has marked as favorite
    # statuses_count the number of tweets
    collection_users.insert_one(u)
    print(u)



if __name__ == '__main__':
    usernames = pandas.read_csv("usernames.csv")
    usernames = usernames.iloc[:1000, 0].tolist()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # get_tweets(usernames[0])
    # get_user(usernames[0])


    for x in usernames:
        try:
            get_tweets(x)
            get_user(x)
        except:
            continue
