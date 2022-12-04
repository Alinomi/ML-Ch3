'''Implementing SVM'''

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
print('Input data size :', X.shape)
print('Output data size :', Y.shape)
print('Label names:', cancer_data.target_names)
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples and {n_neg} negative samples.')

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

'''dealing with more than two classes'''

from sklearn.datasets import load_wine
wine_data = load_breast_cancer()
X = wine_data.data
Y = wine_data.target
print('Input data size :', X.shape)
print('Output data size :', Y.shape)
print('LAbel names:', wine_data.target_names)
n_class0 = (Y == 0).sum()
n_class1 = (Y == 1).sum()
n_class2 = (Y == 2).sum()
print(f'{n_class0} class0 samples, \n{n_class1} class1 samples, \n{n_class2} class2 samples.')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

from sklearn.metrics import classification_report
pred = clf.predict(X_test)
print(classification_report(Y_test, pred))

'''solving linearly non-separable problems with kernels'''

import numpy as np
import matplotlib.pyplot as plt
X = np.c_[
(.3, -.8),
(-1.5, -1),
(-1.3, -.8),
(-1.1, -1.3),
(-1.2, -.3),
(-1.3, -.5),
(-.6, 1.1),
(-1.4, 2.2),
(1, 1),
(1.3, .8),
(1.2, .5),
(.2, -2),
(.5, -2.4),
(.2, -2.3),
(0, -2.7),
(1.3, 2.1)].T
Y = [-1] * 8 + [1] * 8
gamma_option = [1, 2, 4]
for i, gamma in enumerate(gamma_option, 1):
    svm = SVC(kernel='rbf', gamma=gamma)
    svm.fit(X, Y)
    plt.scatter(X[:, 0], X[:, 1], c=['b']*8+['r']*8, zorder=10, cmap=plt.cm.Paired)
    plt.axis('tight')
    XX, YY = np.mgrid[-3:3:200j, -3:3:200j]
    Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    plt.title('gamma = %d' % gamma)
    plt.show()

'''Exploring the face image dataset'''

from sklearn.datasets import fetch_lfw_people
face_data = fetch_lfw_people(min_faces_per_person=80)
X = face_data.data
Y = face_data.target
print('Input data size :', X.shape)
print('Output data size :', Y.shape)
print(('Label names:', face_data.target_names))

for i in range(5):
    print(f'Class {i} has {(Y == i).sum()} samples.')

fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=face_data.target_names[face_data.target[i]])    

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
clf = SVC(class_weight='balanced', random_state=42)
parameters = {'C': [0.1, 1, 10], 'gamma': [1e-07, 1e-08, 1e-06], 'kernel': ['rbf', 'linear']}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, parameters, n_jobs=-1 , cv=5)
grid_search.fit(X_train, Y_train)
print('The best model:\n', grid_search.best_params_)
print('The best averaged performance:', grid_search.best_score_)
clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)
print(f'The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%')
print(classification_report(Y_test, pred, target_names=face_data.target_names))
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(class_weight='balanced', kernel='rbf', random_state=42)

from sklearn.pipeline import Pipeline
model = Pipeline([('pca', pca), ('svc', svc)])
parameters_pipeline = {'svc__C': [1, 3, 10], 'svc__gamma': [0.001, 0.005]}
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)
print('The best model:\n', grid_search.best_params_)
print('The best averaged perfomance:', grid_search.best_score_)
model_best = grid_search.best_estimator_
print(f'The accuracy is : {model_best.score(X_test, Y_test)*100:.1f}%')
pred = model_best.predict(X_test)
print(classification_report(Y_test, pred, target_names=face_data.target_names))

'''Fetal state classification on cardiotocography'''

from collections import Counter
import pandas as pd
df = pd.read_excel('CTG.xls', "Raw Data")
X = df.iloc[1:2126, 3:-2].values
Y = df.iloc[1:2126, -1].values
print(Counter(Y))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)
svc = SVC(kernel='rbf')
parameters = {'C': (100, 1e3, 1e4, 1e5), 'gamma': (1e-08, 1e-7, 1e-6, 1e-5)}
grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
svc_best = grid_search.best_estimator_
accuracy = svc_best.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')
prediction = svc_best.predict(X_test)
report = classification_report(Y_test, prediction)
print(report)