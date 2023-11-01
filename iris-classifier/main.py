import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the iris dataset
iris_data = load_iris()

X = iris_data.data
y = iris_data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# # Create and train the models
# Here is K-Nearest Neighbors model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("KNN accuracy:", knn.score(X_test, y_test))

# Here is Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("LogReg accuracy:", logreg.score(X_test, y_test))

# Here is Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Decision Tree accuracy:", dt.score(X_test, y_test))

# Evaluate metrics - precision, recall, and F1 score
print(classification_report(y_test, knn.predict(X_test)))
print(classification_report(y_test, logreg.predict(X_test)))
print(classification_report(y_test, dt.predict(X_test)))

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

zeros1 = np.zeros_like(xx).ravel()
zeros2 = np.zeros_like(yy).ravel()

X_plot = np.c_[xx.ravel(), yy.ravel(), zeros1, zeros2]

f, axarr = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(10, 3))

for idx, clf, tt in zip([0, 1, 2], [knn, logreg, dt],
                        ['KNN', 'Logistic Regression', 'Decision Tree']):
    Z = clf.predict(X_plot)
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    axarr[idx].set_title(tt)

plt.show()
