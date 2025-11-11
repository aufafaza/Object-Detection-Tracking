import cv2 
from ultralytics import YOLO


def list_cam():
    avail = [] 
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            avail.append(i)
            print(f"Camera found at index {i}")
            cap.release()
        
    return avail
def lookforcamera(): 
    available = list_cam()

    if not available:
        print("No available cameras found.")
        sys.exit()

    print("\nAvailable camera indices:", available)

    while True: 
        try: 
            cam_input = int(input("Select cam: "))
            if cam_input in available: 
                cap = cv2.VideoCapture(cam_input)
                print(f"Using cam at index {cam_input}")
                return cap 

            else:
                print("Invalid idx.")
        except ValueError:
            print("enter a valid number")


model = YOLO("yolo11n.pt")

cap = lookforcamera()

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