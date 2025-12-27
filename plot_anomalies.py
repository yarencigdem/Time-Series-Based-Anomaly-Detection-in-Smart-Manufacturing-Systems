# plot_anomalies_v2.py
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = r"C:\Users\HUAWEI\Desktop\isoforest_results.csv"
OUT_PATH = r"C:\Users\HUAWEI\Desktop\isoforest_pressure_plot_v2.png"

data = pd.read_csv(DATA_PATH)

# datetime'i sırala
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values('datetime')

sensor = 'Pressure'  # dilersen Temperature, Current, Voltage yapabilirsin

# sadece gerekli sütunlar varsa devam et
if sensor not in data.columns:
    raise ValueError(f"'{sensor}' sütunu veri içinde yok. Kolonlar: {list(data.columns)}")

normal = data[data['predicted_anomaly'] == 0]
anomaly = data[data['predicted_anomaly'] == 1]

plt.figure(figsize=(14,6))
plt.scatter(normal['datetime'], normal[sensor], s=10, c='blue', alpha=0.3, label='Normal')
plt.scatter(anomaly['datetime'], anomaly[sensor], s=15, c='red', alpha=0.7, label='Anomali')

plt.title(f"Isolation Forest Anomali Görselleştirme ({sensor})", fontsize=14)
plt.xlabel("Zaman")
plt.ylabel(sensor)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()

print(f"\n✅ Grafik kaydedildi: {OUT_PATH}")
