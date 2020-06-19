import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

myData = pd.read_csv("preprocessed_affection_train.csv")
myData = myData.drop(['Unnamed: 0'], axis=1)

tweets_list = myData.iloc[:, 0].tolist()

myData = pd.read_csv("affection_train.csv")
myData = myData.drop(['Unnamed: 0'], axis=1)
myData = myData.drop(['ID'], axis=1)

sentiment1 = myData.iloc[:, 1:6]
sentiment2 = myData.iloc[:, 9:12]

sentiment = pd.concat([sentiment1,sentiment2], axis=1)

models = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'F1_score', "G mean"])

for i in range(len(tweets_list)):
    tweets_list[i] = re.sub(r'[?|!|\'|"|#|\]|\[|%|,]', r'', tweets_list[i])

# ======================= anger model ===================#

target = sentiment.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)

tfidf = TfidfVectorizer(max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)


estimators = [('SVM', SVC())]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Anger', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("anger.sav", 'wb'))

# ======================= anticipation model ===================#

target = sentiment.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)

tfidf = TfidfVectorizer(max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

bm = LogisticRegression(max_iter=1000)
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Anticipation', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("anticipation.sav", 'wb'))

# ======================= disgust model ===================#

target = sentiment.iloc[:, 2]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)

tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

estimators = [('SVM', SVC())]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Disgust', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("disgust.sav", 'wb'))

# ======================= fear model ===================#

target = sentiment.iloc[:, 3]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)


tfidf = TfidfVectorizer(max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)


estimators = [('SVM', SVC()), ('RF', RandomForestClassifier(n_estimators=20))]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Fear', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("fear.sav", 'wb'))

# ======================= joy model ===================#

target = sentiment.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)


tfidf = TfidfVectorizer(max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)


estimators = [('SVM', SVC()), ('RF', RandomForestClassifier(n_estimators=20))]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Joy', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("joy.sav", 'wb'))

# ======================= sadness model ===================#

target = sentiment.iloc[:, 5]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)


tfidf = TfidfVectorizer(max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

estimators = [('SVM', SVC())]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Sadness', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("sadness.sav", 'wb'))

# ======================= surprise model ===================#

target = sentiment.iloc[:, 6]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)


tfidf = TfidfVectorizer(max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)


estimators = [('SVM', SVC()), ('RF', RandomForestClassifier(n_estimators=20))]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Surprise', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("surprise.sav", 'wb'))

# ======================= trust model ===================#

target = sentiment.iloc[:, 7]

x_train, x_test, y_train, y_test = train_test_split(tweets_list, target, test_size=0.25, random_state=42)

tfidf = TfidfVectorizer(max_df=0.4)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)

estimators = [('SVM', SVC()), ('RF', RandomForestClassifier(n_estimators=20))]
bm = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
bm.fit(x_train, y_train)
scf = bm.predict(x_test)

rec = metrics.recall_score(y_test, scf, average='macro')
pre = metrics.precision_score(y_test, scf, average='macro')
acc = metrics.accuracy_score(y_test, scf)
# gm = metrics.fowlkes_mallows_score(y_test, scf)
f1 = metrics.f1_score(y_test, scf, average='macro')

models = models.append({'model': 'Trust', 'accuracy': acc, 'precision': pre, 'recall': rec, 'F1_score': f1}, ignore_index=True)

pickle.dump(bm, open("trust.sav", 'wb'))
pickle.dump(tfidf, open("vectorizer.pickle", "wb"))
models.to_csv("emotions.csv")

# -----plotting the results------------------#


bars0 = models.iloc[0, 1:5].tolist()
bars1 = models.iloc[1, 1:5].tolist()
bars2 = models.iloc[2, 1:5].tolist()
bars3 = models.iloc[3, 1:5].tolist()
bars4 = models.iloc[4, 1:5].tolist()
bars5 = models.iloc[5, 1:5].tolist()
bars6 = models.iloc[6, 1:5].tolist()
bars7 = models.iloc[6, 1:5].tolist()

barWidth = 1/9

# Set position of bar on X axis
r0 = np.arange(len(bars0))

r1 = [x + barWidth for x in r0]
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]

# Make the plot
plt.bar(r0, bars0, color='#7f6d5f', width=barWidth, edgecolor='white', label='Anger')
plt.bar(r1, bars1, color='yellow', width=barWidth, edgecolor='white', label='Anticipation')
plt.bar(r2, bars2, color='#2d7f5e', width=barWidth, edgecolor='white', label='Disgust')
plt.bar(r3, bars3, color='salmon', width=barWidth, edgecolor='white', label='Fear')
plt.bar(r4, bars4, color='black', width=barWidth, edgecolor='white', label='Joy')
plt.bar(r5, bars5, color='blue', width=barWidth, edgecolor='white', label='Sadness')
plt.bar(r6, bars6, color='red', width=barWidth, edgecolor='white', label='Surprise')
plt.bar(r7, bars7, color='orange', width=barWidth, edgecolor='white', label='Trust')


# Add xticks on the middle of the group bars
plt.xlabel('metrics', fontweight='bold')
plt.xticks([r + barWidth*3 for r in range(len(bars1))], ['Accuracy', 'Precision', 'Recall', 'F1_score'])

# Create legend & Show graphic
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.2))
plt.savefig('emotions.png')
plt.show()
