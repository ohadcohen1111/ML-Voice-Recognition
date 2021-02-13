import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('voice.csv')
# get 0-3 columns in jumps of 2
X = df.iloc[:, 0:20].to_numpy()
y = df.iloc[:, 20].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clf = LogisticRegression(random_state=0, max_iter=3600).fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
print("Score Test: {} ".format(clf.score(X_test, y_test)))
print("Score Train: {} ".format(clf.score(X_train, y_train)))
