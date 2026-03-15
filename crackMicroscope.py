import cv2
from ultralytics import YOLOE  # Note: import YOLOE, not YOLO

# Load the model (make sure it's the right one!)
model = YOLOE("yoloe-11s-seg.pt")  # NOT the -pf version

# THIS IS THE CRITICAL STEP - set your crack detection prompts
crack_prompts = [
 "thin dark cell boundary line",
    "bright intercellular gap",
    "curved line separating skin cells",
    "network of fine dark lines",
    "polygonal cell wall outline",
    "bright ridge between skin cells",
    "dark jagged line pattern",
    "thin branching structure",
    "clear separation between skin layers",
    "irregular cell outline",
    "fine white line dividing tissue",
    "dark filament-like boundary",
    "linear feature in skin microstructure",
    "grid-like pattern of lines",
    "crack-like gap in skin surface",
    "bright edge between cells",
    "thin dark curve in tissue",
    "network of intersecting lines",
    "boundary between epithelial cells",
    "thin white ridge in skin",
    "dark intercellular boundary",
    "bright line between skin cells",
    "thin meandering line",
    "cell perimeter outline",
    "fine dark network pattern"
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