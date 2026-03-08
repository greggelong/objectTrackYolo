import cv2
from ultralytics import YOLO

# Crack model
crack_model = YOLO("yoloe-11s-seg.pt")
crack_prompts = ["pavement crack", "road crack"]

# Person model
person_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOE crack detection
    crack_results = crack_model.predict(frame, prompts=crack_prompts, verbose=False)
    annotated = crack_results[0].plot(boxes=True, masks=True)

    # YOLOv8 person detection
    person_results = person_model(frame)
    annotated = person_results[0].plot(annotated)

    cv2.imshow("Crack + Person Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()