import csv
import process_csv
import pandas as pd

dataset = pd.read_csv("extro-intro-tweets.csv")
# dataset = data.sample(n=10)

index = 0
with open('processing_extro_intro_tweets.csv', 'w', newline='', encoding="utf8") as csvfile:
    fieldnames = ['index', '_id', 'username', 'word', 'emoji', 'hastag', 'len_word', 'len_emoji', 'len_hastag']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    count = 0
    list_with_words = []
    for row in dataset.iterrows():
        # print(row[1].get(key='_id'))
        print(row[1].get(key='username'))
        print(row[1].get(key='_id'))
        id = row[1].get(key='_id')
        username = row[1].get(key='username')
        index += 1
        pro_row = process_csv.preprocessing(row[1].get(key='text'))
        hastags = process_csv.tweet_cleaning(row[1].get(key='text'))[3]
        print(hastags)
        for row in pro_row:
            array1 = pro_row[0]
            array2 = pro_row[1]
            list_with_words.append(array1)
        words = ', '.join(array1)
        emojis = ', '.join(array2)
        if not array2: #if emoji list is empty
            if not hastags:
                writer.writerow({'index': index, '_id': id, 'username': username, 'word': words, 'emoji': 0, 'hastag': 0, 'len_word': len(array1), 'len_emoji': len(array2), 'len_hastag':len(hastags)})
            else:
                writer.writerow({'index': index, '_id': id, 'username': username, 'word': words, 'emoji': 0, 'hastag': hastags, 'len_word': len(array1), 'len_emoji': len(array2), 'len_hastag':len(hastags)})


        else:
            if not hastags:
                writer.writerow({'index': index, '_id': id, 'username': username, 'word': words, 'emoji': emojis, 'hastag': 0, 'len_word': len(array1),
                                 'len_emoji': len(array2), 'len_hastag': len(hastags)})
            else:
                writer.writerow({'index': index, '_id': id, 'username': username, 'word': words, 'emoji': emojis, 'hastag': hastags, 'len_word': len(array1),
                                 'len_emoji': len(array2), 'len_hastag': len(hastags)})

        count = count + 1
        print(count)
    list_with_words.append(array1)

