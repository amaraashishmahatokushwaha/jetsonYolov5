import cv2
import imutils
from yoloDet import YoloTRT

# Initialize the YOLO model
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

# Open the external camera (index 0 is usually the default camera, adjust if necessary)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Resize the frame to a width of 600 pixels
    frame = imutils.resize(frame, width=600)
    
    # Perform inference
    detections, t = model.Inference(frame)
    
    # Display results (optional, uncomment if you want to see FPS)
    # print("FPS: {:.2f} sec".format(1 / t))
    
    # Show the frame with detections
    cv2.imshow("Output", frame)
    
    # Exit if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

