import json, os, time
import requests
import paho.mqtt.client as mqtt

SENSOR_URL = "http://smartcam.uo.ncl.ac.uk:8001/sensor"
CAMERA_URL = "http://smartcam.uo.ncl.ac.uk:8001/camera"
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
PM25_TOPIC = "sensor/pm25"
CAMERA_TOPIC = "camera/raw"
POLL_SECONDS = 5

def get_pm25():
    readings = requests.get(SENSOR_URL, timeout=10).json()
    pm25 = []
    for r in readings:
        if r.get("Variable") != "PM2.5" or r.get("Value") is None:
            continue
        point = {"Timestamp": r["Timestamp"], "Value": r["Value"]}
        if point["Value"] > 50:
            print("Outlier removed:", point, flush=True)
        else:
            pm25.append(point)
    return pm25

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_HOST, 1883)
    client.loop_start()
    while True:
        try:
            pm25 = get_pm25()
            client.publish(PM25_TOPIC, json.dumps(pm25))
            image = requests.get(CAMERA_URL, timeout=10).content
            client.publish(CAMERA_TOPIC, image)
            print(f"Sent {len(pm25)} PM2.5 points and 1 image ({len(image)} bytes)", flush=True)
        except Exception as e:
            print("Error:", e, flush=True)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
