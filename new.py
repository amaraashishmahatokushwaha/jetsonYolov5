import cv2
import imutils
from yoloDet import YoloTRT

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def main():
    # Initialize the YOLO model
    model = YoloTRT(
        library="yolov5/build/libmyplugins.so",
        engine="yolov5/build/yolov5s.engine",
        conf=0.5,
        yolo_ver="v5"
    )
    
    # Set up the GStreamer pipeline (adjust parameters as needed)
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    
    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the original frame for debugging
        #cv2.imshow("Original Frame", frame)
        
        # Resize the frame to a width of 600 pixels
        frame = imutils.resize(frame, width=600)
        
        # Perform inference
        try:
            detections, t = model.Inference(frame)
        except Exception as e:
            print(f"Error during inference: {e}")
            continue
        
        # Display results (optional, uncomment if you want to see FPS)
        # print("FPS: {:.2f} sec".format(1 / t))
        
        # Show the frame with detections (ensure detections are processed correctly)
        cv2.imshow("Output", frame)
        
        # Exit if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

