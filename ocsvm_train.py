# ocsvm_train.py
import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib

# === YOLLAR VE AYARLAR (IF ile aynı) ===
IN_PATH = r"C:\Users\HUAWEI\Desktop\skab_birlesik_temiz.csv"
OUT_DIR = r"C:\Users\HUAWEI\Desktop"
RESULTS_CSV = os.path.join(OUT_DIR, "ocsvm_results.csv")
REPORT_TXT = os.path.join(OUT_DIR, "ocsvm_report.txt")
MODEL_PKL = os.path.join(OUT_DIR, "ocsvm_model.pkl")

# (Ölçekleyiciyi yeniden eğitmiyoruz, IF'te kaydettiğinizi kullanacağız, ancak güvenlik için tekrar yükleyelim)
SCALER_PKL = os.path.join(OUT_DIR, "isoforest_scaler.pkl") 

# ... FEATURE_COLS tanımlaması (önceki koddan) ...
FEATURE_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
    'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS'
]

# === 1) VERİYİ YÜKLE ===
data = pd.read_csv(IN_PATH)
y = data['anomaly'].astype(int)
X = data[FEATURE_COLS].copy()

# === 2) ÖLÇEKLEME (Kaydedilen ölçekleyiciyi kullan) ===
# Bu adım, IF'teki ile aynı ölçekleme dönüşümünü uygular.
try:
    scaler = joblib.load(SCALER_PKL)
    X_scaled = scaler.transform(X)
except:
    # Eğer ölçekleyici kaydı bulunamazsa, yeniden eğit
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)


# === 3) NU TARAMASI (nu = contamination) ===
# IF'deki gibi, gerçek anomali oranı etrafında nu değerlerini tara.
# nu, verinin kaçta kaçının anomali olarak kabul edileceğini belirtir.
nu_values = np.linspace(y.mean() - 0.1, y.mean() + 0.1, 11)
nu_values = sorted(list(set([round(c, 4) for c in nu_values if 0.01 <= c <= 0.5])))

best_nu, best_f1, best_roc_auc, best_preds, best_model = None, -1.0, -1.0, None, None
log_lines = []
log_lines.append(f"Toplam örnek: {len(y)} | Gerçek anomali oranı: {y.mean():.4f}\n")
log_lines.append("Nu (Contam.) ↔ F1-Score | ROC-AUC\n")

for nu in nu_values:
    # One-Class SVM, genellikle 'rbf' çekirdeği (kernel) ile kullanılır.
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
    # ROC-AUC: IF'teki gibi, skorları negatif ile çarpıyoruz.
    roc_auc = roc_auc_score(y, -anomaly_scores) 
    
    log_lines.append(f"  {nu:>5.4f} → F1={f1:.4f} | AUC={roc_auc:.4f}")
    
    if f1 > best_f1:
        best_f1, best_nu, best_preds, best_model, best_roc_auc = f1, nu, preds, model, roc_auc

# ... (Kalan Raporlama ve Kayıt adımları IF koduyla aynıdır, sadece değişken isimleri değişir) ...

# ==============================================================================
# === 4) NİHAİ RAPOR ve KAYIT (Önceki kodun aynısı) ===
# ==============================================================================
report = classification_report(y, best_preds, digits=4)
cm = confusion_matrix(y, best_preds)

print("\n=== OCSVM NİHAİ SONUÇLAR ===")
print(f"En iyi Nu (Contam.): {best_nu}")
print(f"En iyi F1-Score: {best_f1:.4f}")
print(f"ROC-AUC: {best_roc_auc:.4f}")
print("\nClassification Report:\n", report)

# Kayıt işlemleri
out_df = data.copy()
out_df['predicted_anomaly'] = best_preds
out_df.to_csv(RESULTS_CSV, index=False)
joblib.dump(best_model, MODEL_PKL)
# Raporu TXT dosyasına yaz

print("\n✅ OCSVM modelleme tamamlandı.")
# ... Diğer dosya kayıt mesajları ...