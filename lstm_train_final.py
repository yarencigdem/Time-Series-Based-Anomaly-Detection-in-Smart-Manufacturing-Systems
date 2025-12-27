# lstm_train_final.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# GPU ayarı (önceki koddan)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# ==============================================================================
# === YOLLAR VE AYARLAR ===
# ==============================================================================
IN_PATH = r"C:\Users\HUAWEI\Desktop\skab_birlesik_temiz.csv"
OUT_DIR = r"C:\Users\HUAWEI\Desktop"
TIME_STEPS = 30 # Kayan Pencere Boyutu
EPOCHS = 50
BATCH_SIZE = 128
RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE) 

RESULTS_CSV = os.path.join(OUT_DIR, "lstm_results.csv")
REPORT_TXT = os.path.join(OUT_DIR, "lstm_report.txt")
MODEL_PKL = os.path.join(OUT_DIR, "lstm_model.h5") 
SCALER_PKL = os.path.join(OUT_DIR, "isoforest_scaler.pkl") 

FEATURE_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
    'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS'
]

# ==============================================================================
# === 1) VERİ YÜKLEME VE ÖLÇEKLEME ===
# ==============================================================================
data = pd.read_csv(IN_PATH)
y = data['anomaly'].astype(int)
X = data[FEATURE_COLS].copy()

try:
    scaler = joblib.load(SCALER_PKL)
    X_scaled_df = pd.DataFrame(scaler.transform(X), columns=X.columns)
except:
    print("Ölçekleyici yüklenemedi, yeniden eğitiliyor...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==============================================================================
# === 2) 3D ZAMAN SERİSİ DİZİSİ OLUŞTURMA (Predictive Window) ===
# ==============================================================================
def create_predictive_sequences(X, y, time_steps):
    Xs, Ys, ys_labels = [], [], []
    for i in range(len(X) - time_steps):
        # Xs: Pencere (Geçmiş T adımı) -> Model Girdisi
        Xs.append(X.iloc[i:(i + time_steps)].values)
        
        # Ys: Sonraki adım (T+1) -> Modelin Tahmin Etmesi Gereken Çıktı
        Ys.append(X.iloc[i + time_steps].values)
        
        # ys_labels: Sonraki adımın etiketini al (T+1 etiketi)
        ys_labels.append(y.iloc[i + time_steps]) 
        
    return np.array(Xs), np.array(Ys), np.array(ys_labels)

X_seq, Y_seq, y_seq = create_predictive_sequences(X_scaled_df, y, TIME_STEPS)

print(f"3D Girdi Boyutu (X_seq): {X_seq.shape}")
print(f"2D Çıktı Boyutu (Y_seq): {Y_seq.shape}")

# LSTM SADECE NORMAL VERİ üzerinde eğitilir
NORMAL_INDICES = np.where(y_seq == 0)[0]
X_train_normal = X_seq[NORMAL_INDICES]
Y_train_normal = Y_seq[NORMAL_INDICES]

X_test = X_seq
Y_test = Y_seq
y_test = y_seq

print(f"Eğitim Seti Boyutu (Normal Veri): {X_train_normal.shape}")

# ==============================================================================
# === 3) LSTM MODEL MİMARİSİ ===
# ==============================================================================
INPUT_SHAPE = (X_train_normal.shape[1], X_train_normal.shape[2])
OUTPUT_DIM = X_train_normal.shape[2] # 8 özellik

model = Sequential([
    LSTM(128, activation='relu', input_shape=INPUT_SHAPE, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(OUTPUT_DIM) # 8 özellik tahmin et
])

model.compile(optimizer='adam', loss='mae') 
model.summary()

# ==============================================================================
# === 4) MODEL EĞİTİMİ ===
# ==============================================================================
print(f"\nModel {EPOCHS} epoch için eğitiliyor...")
history = model.fit(
    X_train_normal, Y_train_normal, # X_seq -> Y_seq tahmin et
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')]
)

model.save(MODEL_PKL)

# ==============================================================================
# === 5) ANOMALİ TESPİTİ ve EŞİK (THRESHOLD) BELİRLEME ===
# ==============================================================================

# 5a) Tahmin Hatalarını Hesapla
Y_pred = model.predict(X_test)
# Hata = Tahmin edilen Y (Y_pred) ile Gerçek Y (Y_test) arasındaki ortalama mutlak hata (MAE)
errors = np.mean(np.abs(Y_pred - Y_test), axis=1) # Her bir tahmin için 1D hata skoru

# 5b) Eşik Belirleme
NORMAL_TEST_ERRORS = errors[np.where(y_test == 0)[0]]
THRESHOLD_PERCENTILE = 95
threshold = np.percentile(NORMAL_TEST_ERRORS, THRESHOLD_PERCENTILE)

# 5c) İkili Tahmin
preds = (errors > threshold).astype(int) 

# ==============================================================================
# === 6) NİHAİ RAPOR ===
# ==============================================================================

y_test_flat = y_test.flatten()
preds_flat = preds.flatten() 
errors_flat = errors.flatten() 

f1 = f1_score(y_test_flat, preds_flat, pos_label=1)
roc_auc = roc_auc_score(y_test_flat, errors_flat) 

report = classification_report(y_test_flat, preds_flat, digits=4)
cm = confusion_matrix(y_test_flat, preds_flat)

print("\n=== LSTM (Predictive) NİHAİ SONUÇLAR ===")
print(f"Kullanılan Zaman Penceresi (TIME_STEPS): {TIME_STEPS}")
print(f"Eşik (Threshold) Yüzdeliği: {THRESHOLD_PERCENTILE} (Threshold={threshold:.4f})")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# ==============================================================================
# === 7) KAYIT İŞLEMLERİ ===
# ==============================================================================
out_df = data.iloc[TIME_STEPS:].copy() 
out_df['predicted_anomaly'] = preds_flat
out_df.to_csv(RESULTS_CSV, index=False)

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("LSTM (Predictive) Anomali Tespiti - NİHAİ RAPOR\n")
    f.write(f"Zaman Penceresi (TIME_STEPS): {TIME_STEPS}\n")
    f.write(f"Eşik Yüzdeliği: {THRESHOLD_PERCENTILE} (Threshold={threshold:.4f})\n")
    f.write(f"Toplam Test Örneği: {len(y_test_flat)}\n\n")
    f.write("=== Model Performansı ===\n")
    f.write(f"F1-Score (positive=1): {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")
    
print("\n✅ Tüm kayıtlar tamamlandı.")
print(f" - Sonuçlar CSV: {RESULTS_CSV}")
print(f" - Rapor TXT   : {REPORT_TXT}")
print(f" - Model PKL   : {MODEL_PKL}")