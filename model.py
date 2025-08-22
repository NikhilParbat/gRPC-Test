# train_model.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# Save model
joblib.dump(clf, "iris_model.pkl")
