import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('test.csv')
# get 0-3 columns in jumps of 2
X = df.iloc[0:36, 0:20].to_numpy()
y = df.iloc[0:36, 20].to_numpy()


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X
y_train = y
X_test = df.iloc[36:41, 0:20].to_numpy()
y_test = df.iloc[36:41, 20].to_numpy()

print(y_test)
clf = LogisticRegression(random_state=0, max_iter=3600).fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
print("Score Test: {} ".format(clf.score(X_test, y_test)))
print("Score Train: {} ".format(clf.score(X_train, y_train)))
