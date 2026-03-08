import cv2
from ultralytics import YOLOE

model = YOLOE("yoloe-11s-seg.pt")

prompts = ["pavement crack", "road crack", "damaged street", "human", "head"]

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, prompts=prompts, verbose=False)

    annotated = results[0].plot(boxes=True, masks=True)

    cv2.imshow("YOLOE Crack Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()