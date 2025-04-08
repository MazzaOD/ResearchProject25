import cv2 as cv
import numpy as np
import joblib

# ------------------- Load BJJ classifier -------------------
clf = joblib.load("bjj_position_classifier.pkl")  # Trained model

# ------------------- Pose config -------------------
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

width, height = 368, 368
thr = 0.2
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# ------------------- Feedback Logic -------------------

def give_feedback(position):
    feedback_dict = {
        "standing": "Try a takedown or pull guard.",
        "mount": "Maintain pressure and posture!",
        "guard": "Watch out for guard passes.",
        "side control": "Look to escape or recover guard.",
        "back control": "Protect your neck!"
    }
    return feedback_dict.get(position, "")

# ------------------- Helper Functions -------------------

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def analyze_pose(points, frame):
    if points[0] and points[1] and points[8]:
        back_angle = calculate_angle(points[8], points[1], points[0])
        if back_angle < 120:
            cv.putText(frame, "Straighten your back!", (30, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if points[9] and points[8] and points[12]:
        knee_angle = calculate_angle(points[8], points[9], points[12])
        if knee_angle < 160:
            cv.putText(frame, "Knees too bent!", (30, 140),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

def extract_features_from_keypoints(points):
    features = []
    for pt in points[:17]:  # Only use 17 keypoints (ignore Background)
        if pt:
            features.extend(pt[:2])
        else:
            features.extend([0, 0])
    return np.array(features).reshape(1, -1)

# ------------------- Main Pose Detector -------------------

def poseDetector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height),
                (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x, y = (frameWidth * point[0]) / out.shape[3], (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        idFrom, idTo = BODY_PARTS[pair[0]], BODY_PARTS[pair[1]]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.circle(frame, points[idFrom], 3, (0, 0, 255), cv.FILLED)
            cv.circle(frame, points[idTo], 3, (0, 0, 255), cv.FILLED)

    frame = analyze_pose(points, frame)

    # Only try prediction if enough points
    valid_points = [pt for pt in points[:17] if pt is not None]
    if len(valid_points) >= 15:
        features = extract_features_from_keypoints(points)
        try:
            prediction = clf.predict(features)[0]
            cv.putText(frame, f"Position: {prediction}", (30, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            feedback = give_feedback(prediction)
            if feedback:
                cv.putText(frame, f"Feedback: {feedback}", (30, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception as e:
            print("Prediction failed:", e)
    else:
        print("⚠️ Not enough keypoints detected for prediction")

    return frame

# ------------------- Run (Webcam) -------------------

cap = cv.VideoCapture(0)  # use cv.CAP_V4L2 only if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output = poseDetector(frame)
    cv.imshow("Pose Detection (Webcam)", output)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

