# roc_auc_plot.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ==============================================================================
# === YOLLAR VE AYARLAR ===
# ==============================================================================
OUT_DIR = r"C:\Users\HUAWEI\Desktop"
PLOT_NAME = "roc_auc_comparison_bar_chart.png"
OUTPUT_PATH = os.path.join(OUT_DIR, PLOT_NAME)

# ==============================================================================
# === MODEL VERİLERİ (Sizin Nihai Sonuçlarınızdan Alınmıştır) ===
# ==============================================================================
models = ['IF', 'OCSVM', 'LSTM-P', 'LSTM-AE']
# AUC Değerleri: [0.5793, 0.5468, 0.5569, 0.6633]
auc_scores = [0.5793, 0.5468, 0.5569, 0.6633]

# Verileri DataFrame'e dönüştür (Çizim için düzenlemek amacıyla)
df_auc = pd.DataFrame({'Model': models, 'ROC-AUC': auc_scores})
df_auc = df_auc.sort_values(by='ROC-AUC', ascending=False) # En yüksekten en düşüğe sırala

# ==============================================================================
# === ÇİZİM ===
# ==============================================================================

plt.figure(figsize=(10, 6))

# Çubuk renklerini belirleme: En yüksek olanı (LSTM-AE) vurgulamak için kırmızı, diğerleri mavi
colors = ['skyblue' if model != 'LSTM-AE' else 'red' for model in df_auc['Model']]

bars = plt.bar(df_auc['Model'], df_auc['ROC-AUC'], color=colors)


# Değerleri çubukların üzerine yazma ve AE'yi vurgulama
for i, bar in enumerate(bars):
    yval = bar.get_height()
    model_name = df_auc['Model'].iloc[i]
    
    # Yüksek skoru (LSTM-AE) kırmızı ve bold yaz
    if model_name == 'LSTM-AE':
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{round(yval, 4)} (En Yüksek)", 
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='red')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), 
                 ha='center', va='bottom', fontsize=10)

# Başlık ve etiketler
plt.title('Modellerin Anomali Ayırma Potansiyeli (ROC-AUC Karşılaştırması)', fontsize=14, pad=15)
plt.xlabel('Model', fontsize=12)
plt.ylabel('ROC Eğrisi Altındaki Alan (AUC)', fontsize=12)

# Y ekseni sınırlarını belirleme (0.5'ten başlatma)
plt.ylim(0.5, 0.70)
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='Rastgele Tahmin Sınırı (0.5)')

# Legend (Açıklama) ekleme
plt.legend(handles=[bars[df_auc['Model'].tolist().index('LSTM-AE')]], # Kırmızı çubuğu seç
           labels=['LSTM-AE: En Yüksek Potansiyel (0.6633)'],
           loc='upper right', frameon=True, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.5)

# Grafiği kaydet
plt.savefig(OUTPUT_PATH, bbox_inches='tight')

print(f"\n✅ Görselleştirme başarıyla tamamlandı!")
print(f"Dosya yolu: {OUTPUT_PATH}")