import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv('voice.csv')
# get 0-3 columns in jumps of 2
X = df.iloc[:, 0:20].to_numpy()
y = df.iloc[:, 20].to_numpy()

sum_train = 0
sum_test = 0

for i in range(1):

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


    model = SVC(kernel="linear", C=6)
    model.fit(X_train, y_train)
    predict_test = model.predict(X_test)
    predict_train = model.predict(X_train)
    sum_test += accuracy_score(y_test, predict_test)
    sum_train += accuracy_score(y_train, predict_train)
    print("Iteration: {} ".format(i))
print("Accuracy Test: ", sum_test / 1)
print("Accuracy Train: ", sum_train / 1)

# print("Accuracy Train: ", accuracy_score(y_train, predict))

print(model.fit(X_train, y_train))
