import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, ensemble, metrics, preprocessing
from sklearn import naive_bayes, pipeline, feature_extraction


myData = pd.read_csv("extro_intro_dataset2.csv")
x = myData.drop('extrointro', axis=1).values
y = myData['extrointro'].values

# LOGISTIC REGRESSION
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
logreg = LogisticRegression(C=1e5, solver='lbfgs')
logreg.fit(x_train, y_train)
y_predicted = logreg.predict(x_test)

print('LG Recall: %.3f' % metrics.recall_score(y_test, y_predicted, average='macro'))
print('LG Precision: %.3f' % metrics.precision_score(y_test, y_predicted, average='macro'))
print('LG F1-measure: %.3f' % metrics.f1_score(y_test, y_predicted, average='macro'))
print('LG Accuracy: %.3f' % metrics.accuracy_score(y_test, y_predicted))


# SVM
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# minMaxScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# x_train = minMaxScaler.fit_transform(x_train)
# x_test = minMaxScaler.transform(x_test)
#
# model = SVC(C=10, kernel='rbf', gamma=5)
# model.fit(x_train, y_train)
# y_predicted = model.predict(x_test)
#
#
# print('SVM Recall: %.3f' % metrics.recall_score(y_test, y_predicted, average='macro'))
# print('SVM Precision: %.3f' % metrics.precision_score(y_test, y_predicted, average='macro'))
# print('SVM F1-measure: %.3f' % metrics.f1_score(y_test, y_predicted, average='macro'))
# print('SVM Accuracy: %.3f' % metrics.accuracy_score(y_test, y_predicted))


# RANDOM FOREST
# model = ensemble.RandomForestClassifier(criterion='gini', n_estimators=150, max_depth=3)
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)
# model.fit(x_train,y_train)
# y_predicted = model.predict(x_test)
# print('RF Recall: %.3f' % metrics.recall_score(y_test, y_predicted, average='macro'))
# print('RF Precision: %.3f' % metrics.precision_score(y_test, y_predicted, average='macro'))
# print('RF F1-measure: %.3f' % metrics.f1_score(y_test, y_predicted, average='macro'))
# print('RF Accuracy: %.3f' % metrics.accuracy_score(y_test, y_predicted))

# NAIVE BAYES
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)
# gnb = GaussianNB()
# y_predicted = gnb.fit(x_train, y_train).predict(x_test)
# print('NB Recall: %.3f' % metrics.recall_score(y_test, y_predicted, average='macro'))
# print('NB Precision: %.3f' % metrics.precision_score(y_test, y_predicted, average='macro'))
# print('NB F1-measure: %.3f' % metrics.f1_score(y_test, y_predicted, average='macro'))
# print('NB Accuracy: %.3f' % metrics.accuracy_score(y_test, y_predicted))



