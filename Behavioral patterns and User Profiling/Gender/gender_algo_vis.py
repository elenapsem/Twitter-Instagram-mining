import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['Logistic Regre', 'SVM', 'Random Forest', 'Naive Bayes']

Recall = [0.617, 0.595, 0.596, 0.538]
Precision = [0.585, 0.749, 0.686, 0.610]
f1score = [0.562, 0.605, 0.606, 0.525]
Accuracy = [0.597, 0.791, 0.776, 0.754]

x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x-1.5*width, Accuracy, width, label='accuracy')
rects2 = ax.bar(x -0.5*width, Recall, width, label='recall')
rects1 = ax.bar(x + 0.5*width, Precision, width, label='precision')
rects2 = ax.bar(x + 1.5*width, f1score, width, label='f1')

ax.set_ylabel('Measures')
ax.set_title('Evaluation of Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords=None,
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('Evaluation.png')
plt.show()