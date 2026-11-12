import json, os, time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
IN_TOPIC = "camera/raw"
IMAGE_TOPIC = "camera/annotated"
COUNT_TOPIC = "camera/vehicle_count"
OUTPUT_DIR = "/app/output"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
RED = (0, 0, 255)  # OpenCV colours are BGR, so this is red

model = YOLO("yolov8n.pt")
frame_no = 0


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(IN_TOPIC)
    print("Subscribed to", IN_TOPIC, flush=True)


def on_message(client, userdata, msg):
    global frame_no
    frame = cv2.imdecode(np.frombuffer(msg.payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("Could not decode image", flush=True)
        return
    frame_no += 1
    result = model(frame, verbose=False)[0]
    annotated = frame.copy()  # second image, original stays untouched
    count = 0
    for box in result.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue
        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(annotated, VEHICLE_CLASSES[cls], (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    cv2.putText(annotated, f"Vehicles: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    ok, jpg = cv2.imencode(".jpg", annotated)
    if ok:
        client.publish(IMAGE_TOPIC, jpg.tobytes())
    client.publish(COUNT_TOPIC, json.dumps(
        {"frame": frame_no, "count": count, "timestamp": time.time()}))
    cv2.imwrite(f"{OUTPUT_DIR}/latest_annotated.jpg", annotated)
    print(f"Frame {frame_no}: {count} vehicles", flush=True)


os.makedirs(OUTPUT_DIR, exist_ok=True)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, 1883)
client.loop_forever()
