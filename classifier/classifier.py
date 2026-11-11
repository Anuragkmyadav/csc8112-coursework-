import json, os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from ai_edge_litert.interpreter import Interpreter

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
IN_TOPIC = "sensor/pm25"
OUT_TOPIC = "sensor/classified"
MODEL_PATH = "pm25_model.tflite"
OUTPUT_DIR = "/app/output"
LABELS = ["GREEN", "YELLOW", "RED"]
COLORS = {"GREEN": "green", "YELLOW": "gold", "RED": "red"}
PLOT_EVERY = 10

interpreter = Interpreter(model_path=MODEL_PATH)
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
results = {}
batches = 0


def classify(values):
    x = np.array(values, dtype=np.float32).reshape(-1, 1)
    interpreter.resize_tensor_input(inp["index"], x.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    probs = interpreter.get_tensor(out["index"])
    return [LABELS[i] for i in probs.argmax(axis=1)]


def save_charts():
    ts = sorted(results)
    times = [datetime.fromtimestamp(t / 1000) for t in ts]
    values = [results[t][0] for t in ts]
    labels = [results[t][1] for t in ts]
    counts = {l: labels.count(l) for l in LABELS}
    total = len(values)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].bar(LABELS, [counts[l] for l in LABELS], color=[COLORS[l] for l in LABELS])
    ax[0, 0].set_title("Count of each class")
    ax[0, 0].set_ylabel("Readings")
    ax[0, 1].scatter(times, values, c=[COLORS[l] for l in labels], s=8)
    ax[0, 1].set_title("PM2.5 over time (colour = predicted class)")
    ax[0, 1].set_ylabel("PM2.5 (ugm-3)")
    ax[0, 1].tick_params(axis="x", rotation=30)
    ax[1, 0].hist(values, bins=30, color="steelblue")
    ax[1, 0].set_title("PM2.5 value distribution")
    ax[1, 0].set_xlabel("PM2.5 (ugm-3)")
    ax[1, 0].set_ylabel("Frequency")
    summary = "\n".join(
        [f"Total unique readings: {total}"]
        + [f"{l}: {counts[l]} ({100 * counts[l] / total:.1f}%)" for l in LABELS]
        + [f"Min / Mean / Max: {min(values):.2f} / {np.mean(values):.2f} / {max(values):.2f}"])
    ax[1, 1].axis("off")
    ax[1, 1].set_title("Classification summary")
    ax[1, 1].text(0.05, 0.5, summary, fontsize=13, va="center", family="monospace")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/classification_report.png", dpi=120)
    plt.close(fig)
    print("Charts saved.\n" + summary, flush=True)


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(IN_TOPIC)
    print("Subscribed to", IN_TOPIC, flush=True)


def on_message(client, userdata, msg):
    global batches
    points = json.loads(msg.payload)
    if not points:
        return
    labels = classify([p["Value"] for p in points])
    classified = []
    for p, label in zip(points, labels):
        print(f"PM2.5 {p['Value']:.3f} -> {label}", flush=True)
        classified.append({"Timestamp": p["Timestamp"], "Value": p["Value"],
                           "predicted_quality": label})
        results[p["Timestamp"]] = (p["Value"], label)
    client.publish(OUT_TOPIC, json.dumps(classified))
    batches += 1
    if batches % PLOT_EVERY == 0:
        save_charts()


os.makedirs(OUTPUT_DIR, exist_ok=True)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, 1883)
client.loop_forever()
