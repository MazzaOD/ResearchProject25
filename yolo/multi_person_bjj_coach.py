import cv2 as cv
import numpy as np
import joblib

# Load BJJ position classifier
clf = joblib.load("bjj_position_classifier.pkl")

# Load OpenPose model
net_pose = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Load YOLO model for people detection
yolo_net = cv.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
with open("yolo/coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Only detect "person"
person_class_id = class_names.index("person")

# YOLO setup
yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# OpenPose body part config
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

pose_width, pose_height = 368, 368
thr = 0.2

def give_feedback(position):
    return {
        "standing": "Try a takedown or pull guard.",
        "mount": "Maintain pressure and posture!",
        "guard": "Watch out for guard passes.",
        "side control": "Look to escape or recover guard.",
        "back control": "Protect your neck!"
    }.get(position, "")

def extract_keypoints(frame):
    h, w = frame.shape[:2]
    net_pose.setInput(cv.dnn.blobFromImage(frame, 1.0, (pose_width, pose_height),
                                           (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net_pose.forward()[:, :18, :, :]
    points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = int((w * point[0]) / out.shape[3])
        y = int((h * point[1]) / out.shape[2])
        points.append((x, y) if conf > thr else None)

    return points

def extract_features(points):
    features = []
    for pt in points[:17]:
        features.extend(pt[:2] if pt else [0, 0])
    return np.array(features).reshape(1, -1)

def draw_pose(frame, points, offset=(0, 0), label=""):
    ox, oy = offset
    for pair in POSE_PAIRS:
        part_from = BODY_PARTS[pair[0]]
        part_to = BODY_PARTS[pair[1]]
        if points[part_from] and points[part_to]:
            p1 = (points[part_from][0] + ox, points[part_from][1] + oy)
            p2 = (points[part_to][0] + ox, points[part_to][1] + oy)
            cv.line(frame, p1, p2, (0, 255, 0), 2)
            cv.circle(frame, p1, 3, (0, 0, 255), -1)
            cv.circle(frame, p2, 3, (0, 0, 255), -1)

    if label:
        cv.putText(frame, label, (ox + 10, oy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

def detect_people(frame):
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_names = yolo_net.getUnconnectedOutLayersNames()
    outputs = yolo_net.forward(layer_names)

    h, w = frame.shape[:2]
    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == person_class_id and confidence > 0.5:
                center_x, center_y = int(detection[0] * w), int(detection[1] * h)
                bw, bh = int(detection[2] * w), int(detection[3] * h)
                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [boxes[i[0]] for i in indices[:2]]  # Return max 2 persons

# ---------------------- MAIN ----------------------
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    people = detect_people(frame)

    for i, (x, y, w, h) in enumerate(people):
        person_roi = frame[y:y+h, x:x+w]
        if person_roi.size == 0:
            continue

        points = extract_keypoints(person_roi)
        valid_points = [p for p in points[:17] if p is not None]

        if len(valid_points) >= 15:
            features = extract_features(points)
            try:
                prediction = clf.predict(features)[0]
                feedback = give_feedback(prediction)
                label = f"Person {i+1}: {prediction}"
                draw_pose(frame, points, offset=(x, y), label=label)

                if feedback:
                    cv.putText(frame, feedback, (x + 10, y + h - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception as e:
                print(f"[!] Prediction error for person {i+1}:", e)
        else:
            cv.putText(frame, f"Person {i+1}: Pose not clear", (x + 10, y + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.imshow("Multi-Person BJJ Coach", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

