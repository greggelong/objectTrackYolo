import cv2
from ultralytics import YOLO

# Load YOLOE promptable segmentation model
model = YOLO("yoloe-11s-seg.pt")

# Descriptive prompts for pavement cracks
prompts = [
    "thin dark line running across concrete",
    "jagged fracture on asphalt surface",
    "narrow irregular fissure in road",
    "long dark split on pavement",
    "cracked pattern on concrete street",
    "rough uneven line in asphalt",
    "broken section of pavement surface",
    "dark linear break in concrete",
    "narrow damaged road stripe",
    "irregular crack in asphalt road",
    "fractured concrete line",
    "thin black line on street",
    "asphalt surface with jagged splits",
    "small fissures running across road",
    "broken road with uneven gaps",
    "pavement with dark split lines",
    "cracked texture in asphalt surface",
    "linear fracture on street",
    "rough crack pattern on pavement",
    "dark winding fissure in concrete",
    "green"
]

# Open webcam (Mac uses AVFoundation)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOE with low confidence threshold to catch faint cracks
    results = model.predict(frame, prompts=prompts, verbose=False, conf=0.1)

    # Copy frame for annotation
    annotated_frame = frame.copy()

    # Check if any detections exist
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"Detected {len(results[0].boxes)} objects in this frame:")
        for i, box in enumerate(results[0].boxes):
            coords = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
            conf = float(box.conf.cpu().numpy()[0])
            print(f"  Box {i}: {coords}, confidence={conf:.2f}")

        # Draw boxes and masks if desired
        annotated_frame = results[0].plot(boxes=True, masks=True)
    else:
        print("No detections in this frame.")

    # Calculate FPS
    inference_time = results[0].speed["inference"]
    fps = 1000 / inference_time
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Show webcam feed with annotations
    cv2.imshow("YOLOE Crack Detection Debug", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()