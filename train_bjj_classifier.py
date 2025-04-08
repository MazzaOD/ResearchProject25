import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load annotations
with open("/home/mary/Downloads/annotations.json", "r") as f:
    data = json.load(f)


X = []
y = []

for entry in data:
    if "pose2" not in entry or entry["pose2"] is None:
        continue

    keypoints = entry["pose2"]

    flattened = []
    for kp in keypoints:
        if kp and kp[2] > 0.1:
            flattened.extend(kp[:2])
        else:
            flattened.extend([0, 0])

    X.append(flattened)
    y.append(entry["position"])

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print("Model accuracy on test set:", clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))

# Save model
joblib.dump(clf, "bjj_position_classifier.pkl")
print("Model saved to bjj_position_classifier.pkl")


