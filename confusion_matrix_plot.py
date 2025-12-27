# confusion_matrix_plot.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# === YOLLAR VE AYARLAR ===
# ==============================================================================
OUT_DIR = r"C:\Users\HUAWEI\Desktop"
PLOT_NAME = "confusion_matrix_comparison.png"
OUTPUT_PATH = os.path.join(OUT_DIR, PLOT_NAME)

# Confusion Matrix verileri (TN, FP, FN, TP)
# Veriler, daha önceki analizlerinizden alınmıştır.
# Matris yapısı: [[TN, FP], [FN, TP]]

# 1. LSTM Autoencoder (Yüksek AUC, Düşük F1/Recall)
# Kaynak: ae_report.txt -> [[16700, 879], [9090, 422]]
cm_ae = np.array([[16700, 879], [9090, 422]])

# 2. Isolation Forest (Yüksek F1/Recall)
# Kaynak: isoforest_report.txt -> [[10315, 7294], [4583, 4929]]
cm_if = np.array([[10315, 7294], [4583, 4929]])

# Sınıf etiketleri
labels = ['Normal (0)', 'Anomali (1)']

# ==============================================================================
# === GÖRSELLEŞTİRME FONKSİYONU ===
# ==============================================================================

def plot_confusion_matrix(cm, ax, title):
    """Confusion Matrix'i yüzdelik ve sayısal değerlerle birlikte çizen fonksiyon."""
    
    # Gerçek sınıfa göre normalize edilmiş yüzdelikler
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Etiket formatını oluştur (Sayı \n (Yüzde%)
    group_counts = ["{0:,.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm_perc.flatten()]
    
    # Karışık etiket dizisini oluştur (Sayı + Yüzde)
    labels_combined = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels_combined = np.asarray(labels_combined).reshape(2, 2)
    
    # Heatmap çizimi
    sns.heatmap(cm, annot=labels_combined, fmt='', cmap='Blues', ax=ax, 
                cbar=False, annot_kws={"size": 10})
    
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Tahmin Edilen Sınıf', fontsize=10)
    ax.set_ylabel('Gerçek Sınıf', fontsize=10)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9, rotation=90, va="center")

# ==============================================================================
# === ÇİZİMİ GERÇEKLEŞTİR ===
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.3)

# 1. AE Confusion Matrix
plot_confusion_matrix(cm_ae, axes[0], 'A) LSTM Autoencoder - Statik Eşikleme Sorunu')

# 2. IF Confusion Matrix
plot_confusion_matrix(cm_if, axes[1], 'B) Isolation Forest - En İyi Uygulama Dengesi')

# Grafiği kaydet
plt.savefig(OUTPUT_PATH, bbox_inches='tight')

print(f"\n✅ Görselleştirme başarıyla tamamlandı!")
print(f"Dosya yolu: {OUTPUT_PATH}")