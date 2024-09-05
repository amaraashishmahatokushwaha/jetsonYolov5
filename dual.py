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
    
    # Set up the GStreamer pipeline for the CSI camera
    csi_camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not csi_camera.isOpened():
        print("Error: Could not open CSI camera.")
        return
    
    # Set up the USB camera
    usb_camera = cv2.VideoCapture(1)
    if not usb_camera.isOpened():
        print("Error: Could not open USB camera.")
        return
    
    while True:
        # Capture frame from CSI camera
        ret_csi, frame_csi = csi_camera.read()
        if not ret_csi:
            print("Error: Could not read frame from CSI camera.")
            break
        
        # Capture frame from USB camera
        ret_usb, frame_usb = usb_camera.read()
        if not ret_usb:
            print("Error: Could not read frame from USB camera.")
            break
        
        # Resize frames
        frame_csi = imutils.resize(frame_csi, width=600)
        frame_usb = imutils.resize(frame_usb, width=600)
        
        # Perform inference on CSI camera frame
        try:
            detections_csi, t_csi = model.Inference(frame_csi)
        except Exception as e:
            print(f"Error during inference on CSI camera: {e}")
        
        # Perform inference on USB camera frame
        try:
            detections_usb, t_usb = model.Inference(frame_usb)
        except Exception as e:
            print(f"Error during inference on USB camera: {e}")
        
        # Display results
        cv2.imshow("CSI Camera Output", frame_csi)
        cv2.imshow("USB Camera Output", frame_usb)
        
        # Exit if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release captures and close windows
    csi_camera.release()
    usb_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

