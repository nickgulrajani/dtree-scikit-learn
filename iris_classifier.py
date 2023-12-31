import numpy as np
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data   
y = iris.target

# Add feature engineering  
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Models
dt = DecisionTreeClassifier(max_depth=8) 
rf = RandomForestClassifier(n_estimators=100)
svm = SVC()  

models = [dt, rf, svm]

# Evaluate models using cross-validation
for model in models:
  scores = cross_val_score(model, X, y, cv=5)
  print(f"{model.__class__.__name__}: {np.mean(scores):.3f}")  

# Fit best model   
dt.fit(X_train, y_train)

# Evaluate on test data 
acc = dt.score(X_test, y_test)
print(f"Test Accuracy: {acc:.3f}")

# Confusion matrix
y_pred = dt.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred) 

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(conf_mat)

plt.title('Confusion matrix') 
plt.colorbar(im)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
