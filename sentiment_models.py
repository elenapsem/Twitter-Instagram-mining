import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import Preprocessing

# preprocessing of dataset
# myData = pd.read_csv("affection_train.csv")
# myData = myData.drop(['Unnamed: 0'], axis=1)
# myData = myData.drop(['ID'], axis=1)
#
# tweets = myData.iloc[:, 0].tolist()
# sentiment = myData.iloc[:, 7].tolist()
#
# print(pd.Series(sentiment).value_counts())
#
# tweets_list = []
# for i in range(len(tweets)):
#     print(i)
#     x = Preprocessing.preprocessing(tweets[i])
#     tweets_list.append(x[1]+x[2])

# loading the preprocessed dataset to save time
myData = pd.read_csv("preprocessed_affection_train.csv")
myData = myData.drop(['Unnamed: 0'], axis=1)


tweets_list = myData.iloc[:, 0].tolist()
sentiment = myData.iloc[:, 1].tolist()

for i in range(len(tweets_list)):
    tweets_list[i] = re.sub(r'[?|!|\'|"|#|\]|\[|%|,]', r'', tweets_list[i])

x_train, x_test, y_train, y_test = train_test_split(tweets_list, sentiment, test_size=0.25, random_state=42)

# dataframe to store the results of the different models
models = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'F1_score', "G mean"])

tfidf = TfidfVectorizer(ngram_range=(1, 1), max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

# balancing the dataset
oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)

# ----training and evaluating different models----------#
bm = BernoulliNB()
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'BernoulliNB', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm}, ignore_index=True)


bm = RandomForestClassifier(n_estimators=10)
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'RandomForest', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm}, ignore_index=True)

bm = DecisionTreeClassifier()
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'DecisionTree', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm}, ignore_index=True)


bm = LogisticRegression(max_iter=1000)
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Logistic Regression', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm}, ignore_index=True)

bm = SVC()
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'SVM', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm}, ignore_index=True)


estimators = [('SVM', SVC())]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Stacking (SVM,LogReg)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm}, ignore_index=True)


estimators = [('SVM', SVC()), ('RF', RandomForestClassifier(n_estimators=20))]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Stacking (SVM,RF,LogReg)', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1, 'G mean': gm}, ignore_index=True)

models.to_csv("models_sentiment.csv")

# saving the model to load it later and predict sentiment of collected tweets
pickle.dump(bm, open("sentiment_classifier.sav", 'wb'))

# -----plotting the results------------------#


bars0 = models.iloc[0, 1:5].tolist()
bars1 = models.iloc[1, 1:5].tolist()
bars2 = models.iloc[2, 1:5].tolist()
bars3 = models.iloc[3, 1:5].tolist()
bars4 = models.iloc[4, 1:5].tolist()
bars5 = models.iloc[5, 1:5].tolist()
bars6 = models.iloc[6, 1:5].tolist()

barWidth = 1/8

# Set position of bar on X axis
r0 = np.arange(len(bars0))

r1 = [x + barWidth for x in r0]
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]

# Make the plot
plt.bar(r0, bars0, color='#7f6d5f', width=barWidth, edgecolor='white', label='Bernoulli NB')
plt.bar(r1, bars1, color='yellow', width=barWidth, edgecolor='white', label='Random Forest')
plt.bar(r2, bars2, color='#2d7f5e', width=barWidth, edgecolor='white', label='Decision Tree')
plt.bar(r3, bars3, color='salmon', width=barWidth, edgecolor='white', label='Logistic Regression')
plt.bar(r4, bars4, color='black', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r5, bars5, color='blue', width=barWidth, edgecolor='white', label='Stacking (SVM,LogReg)')
plt.bar(r6, bars6, color='red', width=barWidth, edgecolor='white', label='Stacking (SVM,RF,LogReg)')


# Add xticks on the middle of the group bars
plt.xlabel('metrics', fontweight='bold')
plt.xticks([r + barWidth*3 for r in range(len(bars1))], ['Accuracy', 'Precision', 'Recall', 'F1_score'])

# Create legend & Show graphic
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.2))
plt.savefig('sentiment.png')
plt.show()
