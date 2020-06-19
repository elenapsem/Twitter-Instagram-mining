import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * (data[col]/max_val).astype(np.float64))
    data[col + '_cos'] = np.cos(2 * np.pi * (data[col]/max_val).astype(np.float64))
    return data

def create_df(tweets):
    tweets_df = pd.DataFrame(columns=['tweet', 'emojis', 'hashtags', 'sentiment', 'hour', 'day', "followers", "likes"])
    for x in tweets:
        tweets_df = tweets_df.append({'tweet': x['preprocessed_text'][1], 'emojis': x['preprocessed_text'][2],
                                      'hashtags': len(x['preprocessed_text'][3]), 'sentiment': x['sentiment_analysis_supervised'][0],
                                      'hour': x['created'].hour, 'day': x['created'].weekday(), 'followers': x['followers'],
                                      'likes': x['likes']}, ignore_index=True)

    tweets_df = encode(tweets_df, 'hour', 23)
    tweets_df = encode(tweets_df, 'day', 6)

    tweets_df = tweets_df.drop('hour', axis=1)
    tweets_df = tweets_df.drop('day', axis=1)

    one_hot = pd.get_dummies(tweets_df['sentiment'])
    tweets_df = tweets_df.drop('sentiment', axis=1)
    tweets_df = tweets_df.join(one_hot)

    bins = [0, 50, 300, np.inf]
    names = ['small', 'medium', 'high']
    tweets_df['likes'] = pd.cut(tweets_df['likes'], bins, labels=names)

    tweets_df.to_csv('likes_df.csv')

def classification_with_text(filename):
    myData = pd.read_csv("likes_df.csv")
    myData = myData.drop(['Unnamed: 0'], axis=1)

    myData["text"] = (myData["tweet"] + ' ' + myData["emojis"])
    for i in range(len(myData)):
        myData.iloc[i, 11] = re.sub(r'[?|!|\'|"|#|\]|\[|%|,]', r'', myData.iloc[i, 11]).strip()

    myData = myData.loc[:, myData.columns != 'emojis']
    myData = myData.loc[:, myData.columns != 'tweet']
    myData = myData.dropna()

    target = myData['likes']
    myData = myData.loc[:, myData.columns != 'likes']

    x_train, x_test, y_train, y_test = train_test_split(myData, target, test_size=0.25, random_state=42)

    v = TfidfVectorizer()
    x = v.fit_transform(x_train['text'])

    df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
    x_train.drop('text', axis=1, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    x_train.reset_index(drop=True, inplace=True)

    x_train = pd.concat([x_train, df1], axis=1)

    x = v.transform(x_test['text'])
    df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
    x_test.drop('text', axis=1, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    x_test = pd.concat([x_test, df1], axis=1)

    feature_importnace(x_train, y_train, 8, filename)

    models(x_train, x_test, y_train, y_test,filename)



def create_histogram():
    myData = pd.read_csv("likes_df.csv")
    myData = myData.drop(['Unnamed: 0'], axis=1)
    print(myData['likes'].value_counts())
    # plt.hist(myData['likes'], bins=3)
    # plt.show()

def classification_without_text(filename):
    # base model features:
    # day of the week
    # number of followers
    # number of hashtags
    # sentiment
    # hour
    myData = pd.read_csv("likes_df.csv")
    myData = myData.drop(['Unnamed: 0'], axis=1)
    myData = myData.loc[:, myData.columns != 'emojis']
    myData = myData.loc[:, myData.columns != 'tweet']
    myData = myData.dropna()

    target = myData['likes']
    myData = myData.loc[:, myData.columns != 'likes']

    x_train, x_test, y_train, y_test = train_test_split(myData, target, test_size=0.25, random_state=42)

    feature_importnace(myData, target,8,filename)

    models(x_train,x_test,y_train, y_test,filename)

def models(x_train,x_test,y_train, y_test,filename):

    models = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'F1_score', "G mean"])

    bm = BernoulliNB()
    bm.fit(x_train, y_train)
    scf = bm.predict(x_test)

    rec = metrics.recall_score(y_test, scf, average='macro')
    pre = metrics.precision_score(y_test, scf, average='macro')
    acc = metrics.accuracy_score(y_test, scf)
    gm = metrics.fowlkes_mallows_score(y_test, scf)
    f1 = metrics.f1_score(y_test, scf, average='macro')

    models = models.append(
        {'model': 'BernoulliNB', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm},
        ignore_index=True)

    bm = RandomForestClassifier(n_estimators=10)
    bm.fit(x_train, y_train)
    scf = bm.predict(x_test)

    rec = metrics.recall_score(y_test, scf, average='macro')
    pre = metrics.precision_score(y_test, scf, average='macro')
    acc = metrics.accuracy_score(y_test, scf)
    gm = metrics.fowlkes_mallows_score(y_test, scf)
    f1 = metrics.f1_score(y_test, scf, average='macro')

    models = models.append(
        {'model': 'RandomForest', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm},
        ignore_index=True)

    bm = DecisionTreeClassifier()
    bm.fit(x_train, y_train)
    scf = bm.predict(x_test)

    rec = metrics.recall_score(y_test, scf, average='macro')
    pre = metrics.precision_score(y_test, scf, average='macro')
    acc = metrics.accuracy_score(y_test, scf)
    gm = metrics.fowlkes_mallows_score(y_test, scf)
    f1 = metrics.f1_score(y_test, scf, average='macro')

    models = models.append(
        {'model': 'DecisionTree', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm},
        ignore_index=True)

    bm = LogisticRegression(max_iter=1000)
    bm.fit(x_train, y_train)
    scf = bm.predict(x_test)

    rec = metrics.recall_score(y_test, scf, average='macro')
    pre = metrics.precision_score(y_test, scf, average='macro')
    acc = metrics.accuracy_score(y_test, scf)
    gm = metrics.fowlkes_mallows_score(y_test, scf)
    f1 = metrics.f1_score(y_test, scf, average='macro')

    models = models.append(
        {'model': 'Logistic Regression', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1,
         'G mean': gm}, ignore_index=True)

    bm = SVC()
    bm.fit(x_train, y_train)
    scf = bm.predict(x_test)

    rec = metrics.recall_score(y_test, scf, average='macro')
    pre = metrics.precision_score(y_test, scf, average='macro')
    acc = metrics.accuracy_score(y_test, scf)
    gm = metrics.fowlkes_mallows_score(y_test, scf)
    f1 = metrics.f1_score(y_test, scf, average='macro')

    models = models.append(
        {'model': 'SVM', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm},
        ignore_index=True)

    estimators = [('DT', DecisionTreeClassifier()),('NB', BernoulliNB())]
    bm = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=20))
    bm.fit(x_train, y_train)
    scf = bm.predict(x_test)

    rec = metrics.recall_score(y_test, scf, average='macro')
    pre = metrics.precision_score(y_test, scf, average='macro')
    acc = metrics.accuracy_score(y_test, scf)
    gm = metrics.fowlkes_mallows_score(y_test, scf)
    f1 = metrics.f1_score(y_test, scf, average='macro')

    models = models.append(
        {'model': 'Stacking (DT,NB,RF)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1,
         'G mean': gm}, ignore_index=True)

    estimators = [('SVM', SVC()), ('RF', RandomForestClassifier(n_estimators=20))]
    bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    bm.fit(x_train, y_train)
    scf = bm.predict(x_test)


    rec = metrics.recall_score(y_test, scf, average='macro')
    pre = metrics.precision_score(y_test, scf, average='macro')
    acc = metrics.accuracy_score(y_test, scf)
    gm = metrics.fowlkes_mallows_score(y_test, scf)
    f1 = metrics.f1_score(y_test, scf, average='macro')


    models = models.append(
        {'model': 'Stacking (SVM,RF,LogReg)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1,
         'G mean': gm}, ignore_index=True)

    models.to_csv(filename+'.csv')



def feature_importnace(data,target,features,filename):
    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=f_classif, k=features)
    fit = bestfeatures.fit(data, target)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    featureScores = featureScores.nlargest(features, 'Score')
    index = np.arange(len(featureScores['Score']))
    label = featureScores['Specs']
    plt.bar(index, featureScores['Score'])
    plt.xlabel('Feature', fontsize=5)
    plt.ylabel('Importance', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Feature Importance')
    plt.savefig(filename +'_FI.png')
    plt.show()
    #print(featureScores.nlargest(8, 'Score'))  # print 10 best features


def plotting(filename,image):
    models = pd.read_csv(filename)
    models = models.drop(['Unnamed: 0'], axis=1)

    bars0 = models.iloc[0, 1:5].tolist()
    bars1 = models.iloc[1, 1:5].tolist()
    bars2 = models.iloc[2, 1:5].tolist()
    bars3 = models.iloc[3, 1:5].tolist()
    bars4 = models.iloc[4, 1:5].tolist()
    bars5 = models.iloc[5, 1:5].tolist()
    bars6 = models.iloc[6, 1:5].tolist()

    barWidth = 1 / 8

    # Set position of bar on X axis
    r0 = np.arange(len(bars0))

    r1 = [x + barWidth for x in r0]
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]

    # Make the plot
    plt.bar(r0, bars0, color='#7f6d5f', width=barWidth, edgecolor='white', label=models.iloc[0, 0])
    plt.bar(r1, bars1, color='yellow', width=barWidth, edgecolor='white', label=models.iloc[1, 0])
    plt.bar(r2, bars2, color='#2d7f5e', width=barWidth, edgecolor='white', label=models.iloc[2, 0])
    plt.bar(r3, bars3, color='salmon', width=barWidth, edgecolor='white', label=models.iloc[3, 0])
    plt.bar(r4, bars4, color='black', width=barWidth, edgecolor='white', label=models.iloc[4, 0])
    plt.bar(r5, bars5, color='blue', width=barWidth, edgecolor='white', label=models.iloc[5, 0])
    plt.bar(r6, bars6, color='red', width=barWidth, edgecolor='white', label=models.iloc[6, 0])

    # Add xticks on the middle of the group bars
    plt.xlabel('metrics', fontweight='bold')
    plt.xticks([r + barWidth * 3 for r in range(len(bars1))], ['Accuracy', 'Precision', 'Recall', 'F1_score'])

    # Create legend & Show graphic
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.2))
    plt.savefig(image + '.png')
    plt.show()