# ocsvm_train_final.py
import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib

# ==============================================================================
# === YOLLAR VE AYARLAR ===
# ==============================================================================
IN_PATH = r"C:\Users\HUAWEI\Desktop\skab_birlesik_temiz.csv"
OUT_DIR = r"C:\Users\HUAWEI\Desktop"
RANDOM_STATE = 42

RESULTS_CSV = os.path.join(OUT_DIR, "ocsvm_results.csv")
REPORT_TXT = os.path.join(OUT_DIR, "ocsvm_report.txt")
MODEL_PKL = os.path.join(OUT_DIR, "ocsvm_model.pkl")
SCALER_PKL = os.path.join(OUT_DIR, "isoforest_scaler.pkl") # IF'ten kaydedileni kullan

# Sensör kolonları
FEATURE_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
    'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS'
]

# ==============================================================================
# === 1) VERİYİ YÜKLE ===
# ==============================================================================
if not os.path.exists(IN_PATH):
    raise FileNotFoundError(f"HATA: Veri dosyası bulunamadı: {IN_PATH}")

data = pd.read_csv(IN_PATH)
y = data['anomaly'].astype(int)
X = data[FEATURE_COLS].copy()

# ==============================================================================
# === 2) ÖLÇEKLEME ===
# ==============================================================================
try:
    # IF'ten kaydedilen ölçekleyiciyi kullan (tutarlılık için)
    scaler = joblib.load(SCALER_PKL)
    X_scaled = scaler.transform(X)
    print("Ölçekleyici IF'ten yüklendi.")
except:
    # Yüklenemezse yeniden eğit (yedek)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("Ölçekleyici yeniden eğitildi.")

print(f"Toplam örnek sayısı: {len(y)}")
print(f"Gerçek anomali oranı: {y.mean():.4f}")

# ==============================================================================
# === 3) NU TARAMASI (nu = contamination) ===
# ==============================================================================
# Gerçek anomali oranı etrafında nu değerlerini tara
nu_values = np.linspace(y.mean() - 0.1, y.mean() + 0.1, 11)
nu_values = sorted(list(set([round(c, 4) for c in nu_values if 0.01 <= c <= 0.5])))

if not nu_values:
    nu_values = [0.1] # fallback

best_nu, best_f1, best_roc_auc, best_preds, best_model = None, -1.0, -1.0, None, None
log_lines = []
log_lines.append(f"Toplam örnek: {len(y)} | Gerçek anomali oranı: {y.mean():.4f}\n")
log_lines.append("Nu (Contam.) ↔ F1-Score | ROC-AUC\n")

for nu in nu_values:
    # OCSVM modeli
    model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale') 
    
    # Modeli eğit
    model.fit(X_scaled) 
    
    # Anomali skorlarını al (daha büyük skor = normal)
    anomaly_scores = model.decision_function(X_scaled)
    
    # İkili tahminleri al: -1 (anomali) / 1 (normal)
    preds = model.predict(X_scaled)
    preds = (preds == -1).astype(int)  # -1 → anomali(1), 1 → normal(0)
    
    # Metrik hesaplamaları
    f1 = f1_score(y, preds, pos_label=1)
    # ROC-AUC: Skorları negatif ile çarpıyoruz.
    roc_auc = roc_auc_score(y, -anomaly_scores) 
    
    log_lines.append(f"  {nu:>5.4f} → F1={f1:.4f} | AUC={roc_auc:.4f}")
    
    if f1 > best_f1:
        best_f1, best_nu, best_preds, best_model, best_roc_auc = f1, nu, preds, model, roc_auc

print("\n--- Tarama Sonuçları ---")
for line in log_lines[2:]:
    print(line)

# ==============================================================================
# === 4) NİHAİ RAPOR ===
# ==============================================================================
report = classification_report(y, best_preds, digits=4)
cm = confusion_matrix(y, best_preds)

# Konsola yaz
print("\n=== OCSVM NİHAİ SONUÇLAR ===")
print(f"En iyi Nu (Contam.): {best_nu}")
print(f"En iyi F1-Score: {best_f1:.4f}")
print(f"ROC-AUC: {best_roc_auc:.4f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# ==============================================================================
# === 5) ÇIKTILARI KAYDET ===
# ==============================================================================

# 5a) Sonuç CSV
out_df = data.copy()
out_df['predicted_anomaly'] = best_preds
out_df.to_csv(RESULTS_CSV, index=False)

# 5b) Rapor TXT
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("One-Class SVM Anomali Tespiti - NİHAİ RAPOR\n")
    f.write(f"Veri yolu: {IN_PATH}\n")
    f.write(f"Toplam örnek: {len(y)}\n")
    f.write(f"Gerçek anomali oranı: {y.mean():.4f}\n\n")
    f.write("Tarama sonuçları (Nu → F1-Score | ROC-AUC):\n")
    for line in log_lines:
        f.write(line + "\n")
    f.write("\n=== En iyi Model Performansı ===\n")
    f.write(f"Seçilen Nu: {best_nu}\n")
    f.write(f"F1-Score (positive=1): {best_f1:.4f}\n")
    f.write(f"ROC-AUC: {best_roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

# 5c) Modeli kaydet
joblib.dump(best_model, MODEL_PKL)

print("\n✅ Tüm kayıtlar tamamlandı:")
print(f" - Sonuçlar CSV: {RESULTS_CSV}")
print(f" - Rapor TXT   : {REPORT_TXT}")
print(f" - Model PKL   : {MODEL_PKL}")