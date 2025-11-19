import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from logisticRegression.dataPrep import dataPrep
from logisticRegression.logisticRegression import logisticRegression

X_train, X_test, y_train, y_test = dataPrep()

model = logisticRegression(learningRate=0.1, numIterations=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
print("Test accuracy:", acc)

print("Train set size:", len(y_train))
print("Test set size:", len(y_test))
print(confusion_matrix(y_test, y_pred))
print("Positive (1) ratio in train:", np.mean(y_train))
print("Positive (1) ratio in test:", np.mean(y_test))

print(classification_report(y_test, y_pred))


'''

# Example: single-sample prediction from raw inputs
example = {
    "Age": 28,
    "Gender": "Male",
    "Area": "Lalbagh",
    "AreaType": "Undeveloped",
    "HouseType": "Tinshed",
    "District": "Dhaka",
    "NS1": 0,                 # antibody results as in dataset encoding (0/1 or numeric)
    "IgG": 0,
    "IgM": 0,
}
row = makeFeatureRow(metadata, example)
proba = model.predictProb(row)[0]
pred = int(proba >= 0.5)
print("Single-sample prob of Outcome=1:", proba)
print("Single-sample class (threshold 0.5):", pred)

'''



