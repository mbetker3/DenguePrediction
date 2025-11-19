from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def dataPrep():
    root = Path(kagglehub.dataset_download("kawsarahmad/dengue-dataset-bangladesh"))

    #Find CSV files inside that folder
    csvPath = list(root.rglob("*.csv"))[0]  #grabs the first CSV file
    print("Dataset path:", csvPath)

    data = pd.read_csv(csvPath)

    label_encoder = LabelEncoder()

    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Area'] = label_encoder.fit_transform(data['Area'])
    data['AreaType'] = label_encoder.fit_transform(data['AreaType'])
    data['HouseType'] = label_encoder.fit_transform(data['HouseType'])
    data['District'] = label_encoder.fit_transform(data['District'])

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test




