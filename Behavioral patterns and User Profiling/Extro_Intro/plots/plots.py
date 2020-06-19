import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['Logistic Regre', 'SVM', 'Random Forest', 'Naive Bayes']
# Recall = [0.500, 0.305, 0.379, 0.609]
# Precision = [0.515, 0.642, 0.420, 0.618]
# f1score = [0.500, 0.305, 0.379, 0.609]
# Accuracy = [0.522, 0.521, 0.513, 0.517]

Recall = [0.580, 0.585, 0.562, 0.594]
Precision = [0.632, 0.631, 0.843, 0.587]
f1score = [0.574, 0.581, 0.517, 0.558]
Accuracy = [0.677, 0.677, 0.699, 0.560]

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