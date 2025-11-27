from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def dataPrep():
    root = Path(kagglehub.dataset_download("kawsarahmad/dengue-dataset-bangladesh"))
    csvPath = list(root.rglob("*.csv"))[0]
    print("Dataset path:", csvPath)

    data = pd.read_csv(csvPath)
    data = data.dropna(subset=["Outcome"])

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    labelEncoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        labelEncoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y.values,
        test_size=0.2,
        random_state=42
    )

    featureNames = X.columns.tolist()
    metadata = {
        "featureNames": featureNames,
        "scaler": scaler,
        "labelEncoders": labelEncoders
    }

    return X_train, X_test, y_train, y_test, metadata


def makeFeatureRow(rawInputs, metadata):
    featureNames = metadata["featureNames"]
    scaler = metadata["scaler"]
    labelEncoders = metadata["labelEncoders"]

    row = {}
    for feat in featureNames:
        val = rawInputs[feat]
        if feat in labelEncoders:
            le = labelEncoders[feat]
            valStr = str(val)
            if valStr not in le.classes_:
                classes = list(le.classes_)
                classes.append(valStr)
                le.classes_ = np.array(classes)
            encoded = le.transform([valStr])[0]
            row[feat] = encoded
        else:
            row[feat] = float(val)

    dfRow = pd.DataFrame([row], columns=featureNames)
    rowScaled = scaler.transform(dfRow)
    return rowScaled




