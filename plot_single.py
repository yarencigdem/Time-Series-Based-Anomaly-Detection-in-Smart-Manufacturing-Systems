import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\HUAWEI\Desktop\isoforest_results.csv")

sensor = 'Pressure'  # istersen Temperature, Current, Voltage yapabilirsin
src = '8.csv'        # veya '11.csv'

d = df[df['source_file'] == src].copy()
d['datetime'] = pd.to_datetime(d['datetime'])
d = d.sort_values('datetime')

normal = d[d['predicted_anomaly'] == 0]
anomaly = d[d['predicted_anomaly'] == 1]

print(f"{src} -> normal={len(normal)}, anomali={len(anomaly)}")

plt.figure(figsize=(12,5))
plt.scatter(normal['datetime'], normal[sensor], s=15, alpha=0.4, label='Normal', color='blue')
plt.scatter(anomaly['datetime'], anomaly[sensor], s=20, alpha=0.8, label='Anomali', color='red')
plt.title(f"Isolation Forest Anomali Görselleştirme ({src} - {sensor})", fontsize=13)
plt.xlabel("Zaman")
plt.ylabel(sensor)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
