## DecisionTreeClassifier
### Usage
```python
from sklearn.datasets import load_iris
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.20)

clf = DecisionTreeClassifier(max_depth=15)
clf.fit(X, y)

print(f'Accuracy: {sum(clf.predict(X_test) == y_test) / len(y_test):.5f}')
```
Or you can run test.py to compare custom CLF and sklearn CLF:
```bash
python test.py
```
### Contributors
* Oleksii Mytnyk
* Dmytro Mykytenko