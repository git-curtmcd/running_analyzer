import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO("yolov8n-pose.pt")

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Could not find camera. Go buy one broke hoe.")

def calc_angle(a, b, c):
    """Angle at point b, formed by a-b-c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def calc_trunk_lean(shoulder_mid, hip_mid):
    vertical = np.array([0, -1])  # straight up
    torso = shoulder_mid - hip_mid
    cos_angle = np.dot(torso, vertical) / (np.linalg.norm(torso) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

while vc.isOpened():
    rval, frame = vc.read()
    results = model(frame, stream=True)
    for r in results:
        kpts = r.keypoints.xy.cpu().numpy()
        for person in kpts:
            left_knee_angle = calc_angle(person[11], person[13], person[15])
            right_knee_angle = calc_angle(person[12], person[14], person[16])
            kx, ky = int(person[13][0]), int(person[13][1])
            cv2.putText(frame, f"L knee: {left_knee_angle:.0f}", (kx, ky - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            rx, ry = int(person[14][0]), int(person[14][1])
            cv2.putText(frame, f"R knee: {right_knee_angle:.0f}", (rx, ry - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            shoulder_mid = (person[5] + person[6]) / 2
            hip_mid = (person[11] + person[12]) / 2
            trunk_lean = calc_trunk_lean(shoulder_mid, hip_mid)

            tx, ty = int(hip_mid[0]), int(hip_mid[1])
            cv2.putText(frame, f"Trunk: {trunk_lean:.0f}deg", (tx + 10, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
        cv2.imshow("YOLO Live Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vc.release()
cv2.destroyAllWindows()
