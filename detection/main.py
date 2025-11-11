import cv2 
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run detection
    results = model(img)
    
    # Get annotated frame with boxes drawn
    annotated_frame = results[0].plot()
    
    # Show the annotated frame
    cv2.imshow('YOLO Object Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()