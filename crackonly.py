import cv2
from ultralytics import YOLO

# Use the promptable model
model = YOLO("yoloe-11s-seg.pt")

# Prompts focused on cracks
prompts = [
    "pavement crack",
    "road crack",
    "asphalt crack",
    "damaged pavement",
    "human",
    "cup"
]

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        prompts=prompts,
        conf=0.25,
        verbose=False
    )

    annotated = frame.copy()

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            # Draw box
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2)

            # Label
            cv2.putText(
                annotated,
                "CRACK",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,0,255),
                2
            )

    # FPS display
    inference_time = results[0].speed["inference"]
    fps = 1000 / inference_time

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("Crack Detection AI", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()