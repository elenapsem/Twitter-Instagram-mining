import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

dataset = pd.read_csv("extro_intro_dataset.csv")

count = 0
extro = []
intro = []
for value in dataset['extrointro']:
    if value == 0:
        intro.append(value)
    else:
        extro.append(value)
    count = count+1
print(count)
print(len(extro))
print(len(intro))


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'extro', 'intro'
sizes = [56, 37]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('extrointro.png')
plt.show()