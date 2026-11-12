import json, os, queue, threading
import requests
import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
IMAGE_TOPIC = "camera/annotated"
COUNT_TOPIC = "camera/vehicle_count"
PM_TOPIC = "sensor/classified"
DASHBOARD_URL = "http://dashboard.cloud.ncl.ac.uk:8000/api/records"
THRESHOLD = 5

latest_image = None    # F1: latest annotated JPEG bytes
latest_reading = None  # F1: latest classified PM2.5 reading
post_queue = queue.Queue(maxsize=20)


def on_connect(client, userdata, flags, reason_code, properties):
    for topic in (IMAGE_TOPIC, COUNT_TOPIC, PM_TOPIC):
        client.subscribe(topic)
    print("Subscribed to image, count and PM2.5 topics", flush=True)


def on_message(client, userdata, msg):
    global latest_image, latest_reading
    if msg.topic == IMAGE_TOPIC:
        latest_image = msg.payload
    elif msg.topic == PM_TOPIC:
        readings = json.loads(msg.payload)
        if readings:
            latest_reading = max(readings, key=lambda r: r["Timestamp"])
    elif msg.topic == COUNT_TOPIC:
        count = json.loads(msg.payload)["count"]
        if count <= THRESHOLD:  # F2: not heavy traffic -> drop
            print(f"{count} vehicles -> dropped", flush=True)
            return
        if latest_image is None or latest_reading is None:
            print(f"{count} vehicles -> skipped (image or PM2.5 not received yet)", flush=True)
            return
        try:
            post_queue.put_nowait((latest_image, dict(latest_reading), count))
        except queue.Full:
            print("Queue full -> frame dropped", flush=True)


def poster():
    # F3: separate thread sends the POST so MQTT messages keep flowing
    while True:
        image, reading, count = post_queue.get()
        data = {
            "timestamp": int(reading["Timestamp"] // 1000),  # ms -> Unix seconds
            "value": reading["Value"],
            "predicted_quality": reading["predicted_quality"],
        }
        try:
            resp = requests.post(DASHBOARD_URL, data=data,
                                 files={"image": ("frame.jpg", image, "image/jpeg")},
                                 timeout=10)
            print(f"{count} vehicles -> POST {resp.status_code} {data}", flush=True)
        except Exception as e:
            print("POST failed:", e, flush=True)


threading.Thread(target=poster, daemon=True).start()
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, 1883)
client.loop_forever()
