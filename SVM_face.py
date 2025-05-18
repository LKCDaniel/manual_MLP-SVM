import os
import pickle
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


with open(os.path.join('LFW dataset', 'lfw.pkl'), 'rb') as file:
    flw_people = pickle.load(file)  # ['data', 'images', 'target', 'target_names', 'DESCR']



num_class = len(flw_people['target_names'])
instance_num = len(flw_people['target'])

num_per_classes = np.zeros(num_class)
for label in flw_people['target']:
    num_per_classes[label] += 1

histogram = np.zeros(instance_num)
for num_per_class in num_per_classes:
    histogram[int(num_per_class)] += 1

max_frequency = np.max(np.where(histogram != 0))
histogram = histogram[:max_frequency + 1]

print('frequency - classes - percentage - accumulate')
accumulate = 0.0
for i, f in enumerate(histogram):
    if f != 0:
        percentage = f * i * 100 / instance_num
        print(f'{i} - {f:.0f} - {percentage:.2f}% - {100 - accumulate:.2f}%')
        accumulate += percentage

X, Y = [], []
for data, label in zip(flw_people['data'], flw_people['target']):
    if num_per_classes[label] > 100:
        X.append(data)
        Y.append(label)




X, Y = flw_people['data'], flw_people['target']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f'sizes: x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')

model = svm.SVC(kernel='poly', degree=3, coef0=1, C=1.0) # polynomial, rbf
# model = svm.SVC(kernel='rbf', gamma='scale', C=1.0)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nPolynomial kernel')
print(f'accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(f'confusion matrix: {metrics.confusion_matrix(y_test, y_pred)}')

model = svm.SVC(kernel='rbf', ) # polynomial, rbf
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nRBF kernel')
print(f'accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(f'confusion matrix: {metrics.confusion_matrix(y_test, y_pred)}')

plt.figure()
plt.imshow(flw_people.images[0], cmap='gray')

