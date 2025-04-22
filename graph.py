import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load annotations file
with open("/home/mary/Downloads/annotations.json", "r") as f:
    data = json.load(f)

X = []
y = []

# Filter for complete 'pose2' data
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

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load your trained classifier
clf = joblib.load("bjj_position_classifier.pkl")

# Predict
y_pred = clf.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for BJJ Position Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_bjj.png")  # Saves the image
plt.show()

