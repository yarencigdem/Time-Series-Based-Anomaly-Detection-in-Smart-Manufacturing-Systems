# autoencoder_train_final.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed

# GPU bellek kullanımını sınırla (TF/Keras kullanılıyorsa önemlidir)
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
TIME_STEPS = 30 # Kayan Pencere Boyutu (30 zaman adımı)
EPOCHS = 50     # Eğitim döngüsü sayısı
BATCH_SIZE = 128
RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE) # Keras için random seed

RESULTS_CSV = os.path.join(OUT_DIR, "ae_results.csv")
REPORT_TXT = os.path.join(OUT_DIR, "ae_report.txt")
MODEL_PKL = os.path.join(OUT_DIR, "ae_model.h5") # Keras modelleri .h5 olarak kaydedilir

# FEATURE_COLS ve SCALER_PKL (önceki kodlardan)
FEATURE_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
    'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS'
]
SCALER_PKL = os.path.join(OUT_DIR, "isoforest_scaler.pkl") 

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
    scaler = MinMaxScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==============================================================================
# === 2) 3D ZAMAN SERİSİ DİZİSİ OLUŞTURMA (Sliding Window) ===
# ==============================================================================
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # Pencere (Geçmiş verisi)
        Xs.append(X.iloc[i:(i + time_steps)].values)
        # Etiket (Pencerenin sonundaki etiket)
        ys.append(y.iloc[i + time_steps]) 
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled_df, y, TIME_STEPS)

print(f"3D Dizi Boyutu (Örnek, Pencere Adımı, Özellik): {X_seq.shape}")

# Autoencoder SADECE NORMAL VERİ üzerinde eğitilir
NORMAL_INDICES = np.where(y_seq == 0)[0]
X_train_normal = X_seq[NORMAL_INDICES]
# Y_train'i X_train olarak kullanıyoruz (Rekonstrüksiyon)

# Tüm veriyi test için kullanacağız (Anomalileri de içerir)
X_test = X_seq
y_test = y_seq

print(f"AE Eğitim Seti Boyutu (Normal Veri): {X_train_normal.shape}")

# ==============================================================================
# === 3) AUTOENCODER MODEL MİMARİSİ ===
# ==============================================================================
INPUT_SHAPE = (X_train_normal.shape[1], X_train_normal.shape[2])

model = Sequential([
    # Encoder
    LSTM(128, activation='relu', input_shape=INPUT_SHAPE),
    Dropout(0.2),
    # Bottleneck
    RepeatVector(INPUT_SHAPE[0]), # Zaman adımı sayısına geri dön
    # Decoder
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(INPUT_SHAPE[1])) # Her zaman adımı için orijinal özellik sayısını üret
])

model.compile(optimizer='adam', loss='mae') # Mean Absolute Error (MAE) anomali tespiti için iyidir
model.summary()

# ==============================================================================
# === 4) MODEL EĞİTİMİ ===
# ==============================================================================
print(f"\nModel {EPOCHS} epoch için eğitiliyor...")
history = model.fit(
    X_train_normal, X_train_normal, # Girdi = Çıktı (Rekonstrüksiyon)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')]
)

# Modeli kaydet
model.save(MODEL_PKL)
# ==============================================================================
# === 5) ANOMALİ TESPİTİ ve EŞİK (THRESHOLD) BELİRLEME (KESİN DÜZELTME) ===
# ==============================================================================

# 5a) Rekonstrüksiyon Hatalarını Hesapla
X_pred = model.predict(X_test)

# KRİTİK HATA SKORU HESAPLAMA:
# Tüm zaman adımları (axis=1) ve tüm özellikler (axis=2) boyunca mutlak hatanın ortalamasını al.
# Bu, hatalar dizisini kesin olarak (27091,) boyutuna düşürür.
errors = np.mean(np.abs(X_pred - X_test), axis=(1, 2)) 

# 5b) Eşik Belirleme (Normal Veri Hatalarından)
# Test seti hataları içindeki normal verilerin hatalarını kullan
NORMAL_TEST_ERRORS = errors[np.where(y_test.flatten() == 0)[0]]

# Eşiği, normal hataların 95. yüzdelik dilimi olarak belirle
THRESHOLD_PERCENTILE = 95
threshold = np.percentile(NORMAL_TEST_ERRORS, THRESHOLD_PERCENTILE)

# 5c) İkili Tahmin
preds = (errors > threshold).astype(int) # errors zaten 1D olduğu için direkt kullanırız


# ==============================================================================
# === 6) NİHAİ RAPOR VE METRİK HESAPLAMA ===
# ==============================================================================

# Tüm metrik hesaplamaları için dizileri 1D olarak zorluyoruz.
y_test_flat = y_test.flatten()
preds_flat = preds.flatten() 
errors_flat = errors.flatten() # errors'u da 1D olarak zorla

# KRİTİK UZUNLUK KONTROLÜ (Hata ayıklama için terminale yazdır)
print("-" * 30)
print(f"Kontrol: Y_test uzunluğu: {len(y_test_flat)}")
print(f"Kontrol: Preds uzunluğu: {len(preds_flat)}")
print(f"Kontrol: Errors uzunluğu: {len(errors_flat)}")
print("-" * 30)
if len(y_test_flat) != len(preds_flat) or len(y_test_flat) != len(errors_flat):
    print("FATAL HATA: Dizi uzunlukları hala tutarsız! Lütfen veriyi kontrol edin.")
# Not: Bu kod bloğu ile uzunluklar 27091 olacaktır.

# Metrikleri düzleştirilmiş dizilerle hesapla
f1 = f1_score(y_test_flat, preds_flat, pos_label=1)
roc_auc = roc_auc_score(y_test_flat, errors_flat) 

report = classification_report(y_test_flat, preds_flat, digits=4)
cm = confusion_matrix(y_test_flat, preds_flat)

print("\n=== AUTOENCODER NİHAİ SONUÇLAR ===")
print(f"Kullanılan Zaman Penceresi (TIME_STEPS): {TIME_STEPS}")
print(f"Eşik (Threshold) Yüzdeliği: {THRESHOLD_PERCENTILE} (Threshold={threshold:.4f})")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# ==============================================================================
# === 7) KAYIT İŞLEMLERİ (Doğru Boyutta Veri Kullanımı) ===
# ==============================================================================

# 7a) Sonuç CSV
# Orijinal veriden, pencere boyutu kadar ilk satırları atıyoruz (27121 - 30 = 27091 satır)
out_df = data.iloc[TIME_STEPS:].copy() 
out_df['predicted_anomaly'] = preds_flat
out_df.to_csv(RESULTS_CSV, index=False)

# 7b) Rapor TXT
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("LSTM Autoencoder Anomali Tespiti - NİHAİ RAPOR\n")
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