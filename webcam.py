import cv2
from ultralytics import YOLO


model = YOLO("yolov8n-pose.pt")

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Could not find camera. Go buy one broke hoe.")

while vc.isOpened():
    rval, frame = vc.read()
    results = model(frame, stream=True)
    for r in results:
        new_frame = r.plot()
        cv2.imshow("YOLO Live Inference", new_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vc.release()
cv2.destroyAllWindows()
