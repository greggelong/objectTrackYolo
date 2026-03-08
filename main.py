import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("YOLO Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()