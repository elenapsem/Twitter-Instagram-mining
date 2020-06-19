import re
import pandas as pd

file1 = open('affection_train.txt', 'r', encoding='utf-8')
line = file1.readline()
columns = line.split('\t')
columns[-1] = re.sub('\n', '', columns[-1])
data = pd.DataFrame(columns=columns)

while True:
    # Get next line from file
    line = file1.readline()
    # if line is empty
    # end of file is reached
    if not line:
        break
    x = line.split('\t')
    x[-1] = re.sub('\n', '', x[-1])
    data = data.append(pd.Series(x, index=data.columns), ignore_index=True)

file1.close()

data.to_csv('affection_train.csv')
