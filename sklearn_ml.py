import numpy as np
from sklearn import cross_validation
from sklearn import datasets

from sklearn import neighbors
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model


iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
print('K neighbors:',clf.score(x_test,y_test))

clf = svm.SVC()
clf.fit(x_train,y_train)
print('svm.SVC:',clf.score(x_test,y_test))

clf = svm.SVR()
clf.fit(x_train,y_train)
print('svm.SVR:',clf.score(x_test,y_test))

clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
print('DecisionTreeClassifier:',clf.score(x_test,y_test))

clf = LinearRegression()
clf.fit(x_train,y_train)
print('LinearRegression:',clf.score(x_test,y_test))
