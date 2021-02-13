from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('voice.csv')
# get 0-3 columns in jumps of 2
X = df.iloc[:, 0:20].to_numpy()
y = df.iloc[:, 20].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print("Score Test: {} ".format(clf.score(X_test, y_test)))
print("Score Train: {} ".format(clf.score(X_train, y_train)))
