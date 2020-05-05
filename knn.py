"""
KNN stands for K-Nearest Neighbors. KNN is a machine learning algorithm used for classifying data.
Rather than coming up with a numerical prediction such as a students grade or stock price it attempts
to classify data into certain categories

K is an integer value which means no of neighbors it should have.
First, we need to define a K value. This is going to be how many points we should use to make our decision.
The K closest points to the black dot.

Next, we need to find out the class of these K points.

Finally, we determine which class appears the most out of all of our K points and that is our prediction
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing, metrics
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.utils import shuffle
import pickle

data = pd.read_csv("car.data")
# print(data.head(5))
# preprocessing of sklearn helps us to convert non-numeric data into numeric data
# kNN = classification Algorithm where k is an integer value which means amount of neighbour it has.

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
saftey = le.fit_transform(list(data["saftey"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"
X = list(zip(buying, maint, door, persons, lug_boot, saftey))
# print(X)
Y = list(cls)
# print(Y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
# print(x_train)
# print(len(x_train))
# print("-----------")
# print(x_test)
# print(len(x_test))
# print("-----------")
# print(y_train)
# print(len(y_train))
# print("-----------")
# print(y_test)
# print(len(y_test))

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

# OR

# clf = KNeighborsClassifier(n_neighbors=7)
# clf.fit(x_train, y_train)
# print(clf)
#
# y_predict = clf.predict(x_test)
# print(y_predict)
# accuracy = metrics.accuracy_score(y_test, y_predict)
# print(accuracy)


predictions = model.predict(x_test)  # it comes from after training of our data model.fit(x_train, y_train)
# print(predictions)
names = ["unacc", "acc", "good", "vgood"]  # we only use names of ["unacc", "acc", "good", "vgood"]
# bcoz we r predicting class and it has all these value in it

# now we will make predictions on the x_test data
for x in range(len(predictions)):
    # print("predicted: ", predictions[x], "Data: ", x_test[x], "Actual", y_test[x])
    print("predicted:", names[predictions[x]], "Data:", x_test[x], "Actual:", names[y_test[x]])
    # here names is picking the index value of ["unacc", "acc", "good", "vgood"]

    n = model.kneighbors([x_test[x]], 7, True)
    # it will give us the distance B/W the neighbors and index of each neighbor
    print("N: ", n)

style.use("ggplot")
plt.scatter(buying, cls)
plt.xlabel("x")
plt.ylabel("y")
plt.show()