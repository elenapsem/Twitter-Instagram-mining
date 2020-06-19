import re
import json
import pandas as pd
from nltk.stem import WordNetLemmatizer
from urllib.request import urlopen

# read csv file
dataset = pd.read_csv("names.csv")

# ntlk names.csv
x = dataset['name']
print(x)

documents = []
stemmer = WordNetLemmatizer()
for sen in range(0, len(x)):

    document = re.sub(r'\W', ' ', str(x[sen])) # Remove all the special characters
    document = re.sub(r'[.|,|)|(|\|/]', r' ', document) # Removing Punctuations
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document) # remove all single characters
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) # Remove single characters from the start
    document = re.sub('\d+', '', document)  # remove numbers
    document = re.sub(r'\s+', ' ', document, flags=re.I) # Substituting multiple spaces with single space
    cleaner = re.compile('<.*?>')
    document = re.sub(cleaner, ' ', document) # Removing HTML tags
    document = document.lower() # Converting to Lowercase
    document = document.split() # Lemmatization
    document = [stemmer.lemmatize(word) for word in document] #stemming
    document = ' '.join(document)
    documents.append(document)
print(documents)


myKey = "lDzCepQXlQswplGQSy"  # to be obtained from our gender-api account


# Loop through each name, get it's prediction (gender &prediction accuracy) via api and store it against each name in the list
names_list = [] # Define an empty list to store the predictions
for user in documents:
    words = user.split()
    for word in words:
        try:
            url = "https://gender-api.com/get?key=" + myKey + "&name=" + word
            response = urlopen(url)
            decoded = response.read().decode("utf - 8")
            data = json.loads(decoded)
            print(data)
            names_list.append([data["name"], data["gender"], data["accuracy"]])
            names_df = pd.DataFrame(names_list)
            names_df.columns = ["name", "gender_predicted", "accuracy"]
            names_df.to_csv('gender.csv')
            print(names_list)

            # Convert it as dataframe and assign proper variable names
            # names_list.append([data["name"], data["gender"], data["accuracy"]])
        except:
            pass
        if data["name"] != None:
            break

# se kathe epanalipsi twn words na krataei auto pou dn einai unknown
# an ola einai unknown valto unknown