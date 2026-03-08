import cv2
from ultralytics import YOLO

# Load YOLOE prompt-free segmentation model
model = YOLO("yoloe-11s-seg-pf.pt")

# Open webcam (Mac uses AVFoundation)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOE
    results = model.predict(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot(boxes=True, masks=False)

    # Calculate FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time

    text = f"FPS: {fps:.1f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]

    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10

    cv2.putText(
        annotated_frame,
        text,
        (text_x, text_y),
        font,
        1,
        (255,255,255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("YOLOE Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()