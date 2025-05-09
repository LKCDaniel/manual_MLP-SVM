import os
import pickle
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


with open(os.path.join('LFW dataset', 'lfw.pkl'), 'rb') as file:
    flw_people = pickle.load(file)  # ['data', 'images', 'target', 'target_names', 'DESCR']

X, Y = flw_people['data'], flw_people['target']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f'sizes: x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')

model = svm.SVC(kernel='poly') # polynomial, rbf
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nPolynomial kernel')
print(f'accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(f'confusion matrix: {metrics.confusion_matrix(y_test, y_pred)}')

model = svm.SVC(kernel='rbf') # polynomial, rbf
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nRBF kernel')
print(f'accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(f'confusion matrix: {metrics.confusion_matrix(y_test, y_pred)}')

plt.figure()
plt.imshow(flw_people.images[0], cmap='gray')

# "C:\Program Files\Python312\python.exe" "C:\Users\18611\Desktop\Final Assessment\SVM_face.py"
# sizes: x_train: (10586, 2914), y_train: (10586,), x_test: (2647, 2914), y_test: (2647,)
#
# Polynomial kernel
# accuracy: 0.16093690970910465
# confusion matrix: [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
#
# RBF kernel
# accuracy: 0.11371363808084624
# confusion matrix: [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
#
# Process finished with exit code 0
