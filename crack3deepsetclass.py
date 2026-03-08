import cv2
from ultralytics import YOLOE  # Note: import YOLOE, not YOLO

# Load the model (make sure it's the right one!)
model = YOLOE("yoloe-11s-seg.pt")  # NOT the -pf version

# THIS IS THE CRITICAL STEP - set your crack detection prompts
crack_prompts = [
    "pavement crack", 
    "road fissure", 
    "concrete fracture",
    "asphalt crack",
    "thin dark line on road",
    "human person",
    "art worker with green vest yellow hard hat"
]
model.set_classes(crack_prompts)

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference - now it knows what to look for!
    results = model.predict(frame, conf=0.1, verbose=False)
    
    # Annotate and display
    annotated = results[0].plot()
    cv2.imshow("YOLOE Crack Detection", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()