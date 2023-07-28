import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
url = "C:\\Users\\ludwi\\PycharmProjects\\MLGithub\\mushroom\\agaricus-lepiota.data"
names1 = ["Edible", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", "gill-spacing"
         , "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring"
         , "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type"
         , "spore-print-color", "population", "habitat"]
dataset = pd.read_csv(url, names=names1)
Tdataset = np.transpose(dataset.values.tolist())
for i in range(0, 23):
    le = LabelEncoder()
    Tdataset[i] = le.fit_transform(Tdataset[i])
DataValues = np.transpose(Tdataset)
dataset = pd.DataFrame(data=DataValues, columns=names1)
dataset = dataset.astype("int")
y = dataset.iloc[:, :1].values
X = dataset.iloc[:, 1:].values
X = np.asarray(X)
y = np.asarray(y)
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print("Accuracy: %.2f%%" % (acc * 100.0))
print(confusion_matrix(y_test, y_pred))







