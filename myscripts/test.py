import cv2 as cv
import numpy as np

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
thr = 0.2  # Confidence threshold

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

def calculate_angle(a, b, c):
    """Calculate the angle between three keypoints."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def analyze_pose(points, frame):
    """Analyze the pose and overlay feedback on the frame."""
    # Example: Analyze back posture
    if points[1] and points[8]:  # Neck and Hip exist
        back_angle = calculate_angle(points[8], points[1], points[0])  # Hip-Neck-Head
        if back_angle < 120:
            cv.putText(frame, "Straighten your back!", (50, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Example: Analyze knee angle (this is for a squat-like position)
    if points[9] and points[8] and points[12]:  # Right knee, hip, and left knee
        knee_angle = calculate_angle(points[8], points[9], points[12])  # Hip-RKnee-LKnee
        if knee_angle < 160:
            cv.putText(frame, "Knees too bent! Keep them aligned.", (50, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

def poseDetector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
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
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    
    # Perform the analysis on detected points and give feedback
    frame = analyze_pose(points, frame)

    return frame

# Choose video source (0 for webcam, or provide video file path)
video_source = '/home/mary/Downloads/video.mp4'  # Change this to a filename for video processing
cap = cv.VideoCapture(0, cv.CAP_V4L2)  # Explicitly use V4L2 backend


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output = poseDetector(frame)
    cv.imshow("Pose Detection", output)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

