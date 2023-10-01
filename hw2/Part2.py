import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

data = pd.read_csv('datasets/politeness_data.csv')
data.info()



# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# clf = LogisticRegression(lr=0.01)
# print('X_train', len(X_train))
# print('y_train', len(y_train))
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# def accuracy(y_pred, y_test):
#     return np.sum(y_pred==y_test)/len(y_test)

# acc = accuracy(y_pred, y_test)
# print(acc)
