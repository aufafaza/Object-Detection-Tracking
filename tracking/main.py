import cv2 
import sys 

from cv2 import Tracker



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




tracker_type = 'BOOSTING'
tracker = cv2.legacy.TrackerBoosting_create()



video = lookforcamera()
video.set(3, 640)
video.set(4, 480)

if not video.isOpened():
    print("Could not open video")
    sys.exit() 


ok, frame = video.read() 

if not ok: 
    print("Cannot read video file")
    sys.exit() 


bbox = (287, 23, 86, 320)

bbox = cv2.selectROI(frame, False)
ok = tracker.init(frame, bbox) 

while True: 
    ok, frame = video.read() 
    if not ok: 
        break 

    timer = cv2.getTickCount() 

    ok, bbox = tracker.update(frame) 

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) 

    if ok: 
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

    else:
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)

    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2) 

    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break 






