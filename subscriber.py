import paho.mqtt.client as mqtt
import cv2
import subprocess
import signal
import time
import threading 

camera = None
camera_on = False
object_detection_process = None
object_detection_running = False
running = True

broker_address = "192.168.0.177" 
port = 1883


def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! appsink"
    )

def start_camera():
    global camera, camera_on
    if not camera_on:
        pipeline = gstreamer_pipeline()
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not camera.isOpened():
            print("Camera failed to open with GStreamer pipeline!")
            return
        print("Camera is ON")
        camera_on = True
        threading.Thread(target=camera_loop).start()  
def camera_loop():
    global camera, camera_on
    try:
        while camera_on:
            ret, frame = camera.read()
            if not ret:
                print("Camera failed to read frame!")
                break
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Exception occurred while reading frame: {e}")
    finally:
        stop_camera()

def stop_camera():
    global camera, camera_on
    if camera_on and camera is not None:
        try:
            camera.release()
            cv2.destroyAllWindows()
            camera = None
            camera_on = False
            print("Camera is OFF")
        except Exception as e:
            print(f"Exception occurred while releasing camera: {e}")

def run_object_detection():
    global object_detection_running, object_detection_process
    if not object_detection_running:
        print("Running object detection")
        object_detection_process = subprocess.Popen(
            ['python3', '/home/jetson/JetsonYolov5/new.py'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        object_detection_running = True
    else:
        print("Object detection is already running")

def stop_object_detection():
    global object_detection_running, object_detection_process
    if object_detection_running and object_detection_process:
        print("Stopping object detection...")
        object_detection_process.terminate()
        object_detection_process.wait()
        object_detection_running = False
    else:
        print("Object detection is not running")

def handle_object_detection(action):
    if action == "start":
        run_object_detection()
    elif action == "stop":
        stop_object_detection()
    else:
        print(f"Unknown action: {action}")

def on_message(client, userdata, message):
    command = message.payload.decode("utf-8").strip().lower()
    print(f"Received command: {command}")

    if command == "camera_on":
        start_camera()
    elif command == "camera_off":
        stop_camera()
    elif command == "run_detection":
        handle_object_detection("start")
    elif command == "stop_detection":
        handle_object_detection("stop")
    else:
        print(f"Unknown command: {command}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe("jetson/commands",1)
    else:
        print(f"Failed to connect, return code {rc}")

client = mqtt.Client("remote_controller")
client.clean_session=False

def connect_mqtt():
    try:
        client.connect(broker_address, port)
        print("Connected to MQTT broker")
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")

def signal_handler(sig, frame):
    global running
    print("Shutting down gracefully...")
    running = False
    stop_camera()
    stop_object_detection()
    client.loop_stop()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

client.on_message = on_message
client.on_connect = on_connect

connect_mqtt()
client.loop_start()

try:
    while running:
        time.sleep(0.1)  
finally:
    stop_camera()
    stop_object_detection()
    client.loop_stop()
    print("Script terminated.")

