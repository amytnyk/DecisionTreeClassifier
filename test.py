from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import timeit
from DecisionTreeClassifier import DecisionTreeClassifier as CustomDecisionTreeClassifier, LeafNode

iris = load_iris()
X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.20)

classifiers = {"Sklearn": DecisionTreeClassifier(),
               "Custom": CustomDecisionTreeClassifier(10)}


def fit_clf(clf_id: str):
    def fit():
        classifiers[clf_id] = classifiers[clf_id].fit(X, y)
    return fit


def get_clf_accuracy(clf_id: str):
    return sum(classifiers[clf_id].predict(X_test) == y_test) / len(y_test)


def test_classifiers():
    for clf_id in classifiers:
        time_taken = timeit.timeit(fit_clf(clf_id), number=1)
        accuracy = get_clf_accuracy(clf_id)
        print(f"{clf_id} classifier:\nTime taken to train: {time_taken}s\nAccuracy: {accuracy}\n")


if __name__ == "__main__":
    test_classifiers()
