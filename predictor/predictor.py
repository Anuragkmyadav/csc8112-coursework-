import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from ml_engine import MLPredictor

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
IN_TOPIC = "sensor/pm25"
OUTPUT_DIR = "/app/output"
TARGET = 8706      # unique PM2.5 points needed
STALE_LIMIT = 40   # messages in a row with nothing new = full data loop already seen

unique = {}        # Timestamp -> Value (a dict removes duplicates automatically)
received = 0
stale = 0


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(IN_TOPIC)
    print("Subscribed to", IN_TOPIC, flush=True)


def on_message(client, userdata, msg):
    global received, stale
    batch = json.loads(msg.payload)
    before = len(unique)
    for p in batch:
        unique[p["Timestamp"]] = p["Value"]
    received += len(batch)
    stale = stale + 1 if len(unique) == before else 0
    print(f"Received {received} points, {len(unique)} unique (target {TARGET})", flush=True)
    if len(unique) >= TARGET or stale >= STALE_LIMIT:
        client.disconnect()


def process():
    # (b) remove duplicates + sort by time
    df = pd.DataFrame(sorted(unique.items()), columns=["Timestamp", "Value"])
    print(f"After removing duplicates and sorting: {len(df)} points", flush=True)
    print("Last 2 values:", df["Value"].tail(2).tolist(), flush=True)

    # (c) 10-minute average, (d) readable date-time
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    avg = df.set_index("Timestamp").resample("10min").mean().dropna().reset_index()
    readable = avg.assign(Timestamp=avg["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    print("PM2.5 averaged over 10 minutes:", flush=True)
    print(readable.to_string(index=False), flush=True)

    # (e) line chart of the averaged data
    plt.figure(figsize=(12, 5))
    plt.plot(avg["Timestamp"], avg["Value"], marker=".", linewidth=1)
    plt.title("PM2.5 averaged over 10-minute periods")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (ugm-3)")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pm25_10min_average.png", dpi=120)
    plt.close()

    # (f) Prophet prediction with the provided ML engine, (i) save its graph
    predictor = MLPredictor(avg.copy())
    predictor.train()
    forecast = predictor.predict()
    fig = predictor.plot_result(forecast)
    fig.savefig(f"{OUTPUT_DIR}/pm25_prediction.png", dpi=120)
    try:
        print("Predicted PM2.5 (next periods):", flush=True)
        print(forecast[["ds", "yhat"]].tail(15).to_string(index=False), flush=True)
    except Exception:
        print(forecast.tail(15), flush=True)
    print("Done. Graphs saved in", OUTPUT_DIR, flush=True)


os.makedirs(OUTPUT_DIR, exist_ok=True)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, 1883)
client.loop_forever()   # returns after client.disconnect()
process()
