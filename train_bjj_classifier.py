import json
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
        if kp and kp[2] > 0.1:  # confidence threshold
            flattened.extend(kp[:2])
        else:
            flattened.extend([0, 0])

    X.append(flattened)
    y.append(entry["position"])

X = np.array(X)
y = np.array(y)

print("Class distribution:", Counter(y))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier with class balancing
clf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print("Model accuracy on test set:", clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))

# Save model and scaler
joblib.dump(clf, "bjj_position_classifier2.pkl")
joblib.dump(scaler, "bjj_scaler.pkl")
print("Model and scaler saved.")

