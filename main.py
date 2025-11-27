import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from dataPrep import dataPrep
from dataPrep import makeFeatureRow
from logisticRegression import logisticRegression

def trainModel():
    X_train, X_test, y_train, y_test, metadata = dataPrep()
    model = logisticRegression(learningRate=0.05, numIterations=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print("Test accuracy:", acc)
    return model, metadata

def getPatientProfile():
    print("Enter patient information for dengue risk prediction.")
    age = float(input("Age (years): "))
    ns1 = int(input("NS1 test (1 = positive, 0 = negative): "))
    igg = int(input("IgG test (1 = positive, 0 = negative): "))
    igm = int(input("IgM test (1 = positive, 0 = negative): "))

    gender = input("Gender (Male/Female): ").strip().title()
    area = input("Area (name of area: Mirpur, Chawkbazar, Motijheel): ").strip()
    area_type = input("Area type (Developed/Undeveloped): ").strip().title()
    house_type = input("House type (Building, Tinshed, Other): ").strip()
    district = input("District (Dhaka): ").strip()

    return {
        "Age": age,
        "NS1": ns1,
        "IgG": igg,
        "IgM": igm,
        "Gender": gender,
        "Area": area,
        "AreaType": area_type,
        "HouseType": house_type,
        "District": district
    }

def riskLabel(prob):
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    return "LOW"

def dengueRiskPredict():
    model, metadata = trainModel()
    print("\nModel trained. Now you can enter patient profiles.\n")

    while True:
        profile = getPatientProfile()
        x_row = makeFeatureRow(profile, metadata)
        prob = float(model.predictProb(x_row)[0])
        label = riskLabel(prob)

        print(f"\nEstimated probability of dengue: {prob:.2%}")
        print(f"Risk level: {label}\n")

        again = input("Predict for another person? (y/n): ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    dengueRiskPredict()



