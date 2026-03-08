import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOE model
model = YOLO("yoloe-11s-seg.pt")

# Print what classes YOLOE actually knows
print("YOLOE knows these classes:")
for i, class_name in model.names.items():
    print(f"  {i}: {class_name}")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOE doesn't use prompts - just run normal inference
    results = model.predict(frame, conf=0.25, verbose=False)

    # Check detections
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"Detected {len(results[0].boxes)} objects")
        
        # Print what was detected
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  - {model.names[class_id]}: {conf:.2f}")
    else:
        print("No detections")

    # Annotate and display
    annotated_frame = results[0].plot(boxes=True, masks=True)
    
    # Add FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time if inference_time > 0 else 0
    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("YOLOE Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()