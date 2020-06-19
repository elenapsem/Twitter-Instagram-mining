import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("users_dataset.csv")

gender = []
for i in range(len(dataset)):
    gender.append(dataset.iloc[i, 8])
print(gender)

female = []
male = []
for name in gender:
    print(name)
    if name == 'female':
        female.append(name)
    if name == 'male':
        male.append(name)

print(len(female))
print(len(male))

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'male', 'female', 'unknown'
sizes = [538, 207, 226]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()