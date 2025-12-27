# isoforest_train_final.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib

# ==============================================================================
# === YOLLAR VE AYARLAR ===
# ==============================================================================
IN_PATH = r"C:\Users\HUAWEI\Desktop\skab_birlesik_temiz.csv"  # Birleştirdiğiniz veri yolu
OUT_DIR = r"C:\Users\HUAWEI\Desktop"                           # Çıktılar masaüstüne
RANDOM_STATE = 42                                            # Tekrarlanabilirlik için

RESULTS_CSV = os.path.join(OUT_DIR, "isoforest_results.csv")
REPORT_TXT = os.path.join(OUT_DIR, "isoforest_report.txt")
MODEL_PKL = os.path.join(OUT_DIR, "isoforest_model.pkl")
SCALER_PKL = os.path.join(OUT_DIR, "isoforest_scaler.pkl") # Ölçekleyiciyi kaydetmek

# Sensör kolonları (SKAB verisine özgü olarak varsayılmıştır)
FEATURE_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
    'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS'
]

# ==============================================================================
# === 1) VERİYİ YÜKLE VE HAZIRLA ===
# ==============================================================================
if not os.path.exists(IN_PATH):
    raise FileNotFoundError(f"HATA: Veri dosyası bulunamadı: {IN_PATH}")

data = pd.read_csv(IN_PATH)

# Güvenlik: Eksik sensör kolonu var mı?
missing = [c for c in FEATURE_COLS if c not in data.columns]
if missing:
    raise ValueError(f"HATA: Eksik sensör kolonu(lar): {missing}.")

# Hedef etiket
if 'anomaly' not in data.columns:
    raise ValueError("HATA: CSV içinde 'anomaly' kolonu bulunamadı.")
y = data['anomaly'].astype(int)

# Veri setini ayır
X = data[FEATURE_COLS].copy()

# ==============================================================================
# === 2) ÖLÇEKLEME (Min-Max) ===
# ==============================================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Toplam örnek sayısı: {len(y)}")
print(f"Gerçek anomali oranı: {y.mean():.4f}")

# ==============================================================================
# === 3) CONTAMINATION OPTİMİZASYONU (En iyi F1 için) ===
# ==============================================================================
# Anomali tespiti, yarı-denetimsiz bir yöntem olduğundan, en iyi eşiği bulmak için 
# contamination oranını tarıyoruz.
contam_values = np.linspace(y.mean() - 0.1, y.mean() + 0.1, 11) # Gerçek oranın etrafında dene
contam_values = [c for c in contam_values if 0.01 <= c <= 0.5] # Makul sınırlar

# Eğer veride hiç anomali yoksa (0), varsayılan bir aralık dene
if y.mean() < 0.01:
    contam_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    
# Tekrar eden değerleri kaldırıp sırala
contam_values = sorted(list(set([round(c, 4) for c in contam_values])))
if not contam_values:
    contam_values = [0.1] # fallback

best_c, best_f1, best_roc_auc, best_preds, best_model = None, -1.0, -1.0, None, None
log_lines = []
log_lines.append(f"Toplam örnek: {len(y)} | Gerçek anomali oranı: {y.mean():.4f}\n")
log_lines.append("Contamination ↔ F1-Score | ROC-AUC\n")

for c in contam_values:
    model = IsolationForest(
        contamination=c,
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Modeli tüm veri üzerinde eğit (Unsupervised/Semi-Supervised yaklaşım)
    model.fit(X_scaled) 
    
    # Anomali skorlarını al
    anomaly_scores = model.decision_function(X_scaled)
    
    # İkili tahminleri al: -1 (anomali) / 1 (normal)
    preds = model.predict(X_scaled)
    preds = (preds == -1).astype(int)  # -1 → anomali(1), 1 → normal(0)
    
    # Metrik hesaplamaları
    f1 = f1_score(y, preds, pos_label=1)
    # ROC-AUC için: decision_function değerleri küçüldükçe anomali olur.
    # ROC-AUC'nin doğru çalışması için yüksek skorlar pozitif sınıfa (anomali) ait olmalıdır.
    # Bu yüzden skorları negatif ile çarparız.
    roc_auc = roc_auc_score(y, -anomaly_scores) 
    
    log_lines.append(f"  {c:>5.4f} → F1={f1:.4f} | AUC={roc_auc:.4f}")
    
    if f1 > best_f1:
        best_f1, best_c, best_preds, best_model, best_roc_auc = f1, c, preds, model, roc_auc

print("\n--- Tarama Sonuçları ---")
for line in log_lines[2:]:
    print(line)

# ==============================================================================
# === 4) NİHAİ RAPOR ===
# ==============================================================================
report = classification_report(y, best_preds, digits=4)
cm = confusion_matrix(y, best_preds)

# Konsola kısa özet
print("\n=== NİHAİ SONUÇLAR ===")
print(f"En iyi Contamination: {best_c}")
print(f"En iyi F1-Score: {best_f1:.4f}")
print(f"ROC-AUC: {best_roc_auc:.4f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# ==============================================================================
# === 5) ÇIKTILARI KAYDET ===
# ==============================================================================

# 5a) Sonuç CSV (gerçek vs tahmin)
out_df = data.copy()
out_df['predicted_anomaly'] = best_preds
out_df.to_csv(RESULTS_CSV, index=False)

# 5b) Rapor TXT
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("Isolation Forest Anomali Tespiti - NİHAİ RAPOR\n")
    f.write(f"Veri yolu: {IN_PATH}\n")
    f.write(f"Toplam örnek: {len(y)}\n")
    f.write(f"Gerçek anomali oranı: {y.mean():.4f}\n\n")
    f.write("Tarama sonuçları (contamination → F1-Score | ROC-AUC):\n")
    for line in log_lines:
        f.write(line + "\n")
    f.write("\n=== En iyi Model Performansı ===\n")
    f.write(f"Seçilen Contamination: {best_c}\n")
    f.write(f"F1-Score (positive=1): {best_f1:.4f}\n")
    f.write(f"ROC-AUC: {best_roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

# 5c) Modeli kaydet
joblib.dump(best_model, MODEL_PKL)

# 5d) Ölçekleyiciyi kaydet (Gelecek tahminler için kritik)
joblib.dump(scaler, SCALER_PKL)

print("\n✅ Tüm kayıtlar tamamlandı:")
print(f" - Sonuçlar CSV: {RESULTS_CSV}")
print(f" - Rapor TXT   : {REPORT_TXT}")
print(f" - Model PKL   : {MODEL_PKL}")
print(f" - Ölçekleyici : {SCALER_PKL}")