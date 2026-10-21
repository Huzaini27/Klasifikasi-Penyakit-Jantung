import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TAHAP 1: PERSIAPAN & PRE-PROCESSING DATA
print("=== MEMBACA DATASET ===")
data = pd.read_csv('heart.csv')
print(data.head())

# Pembersihan Data
data = data.dropna()
data = data.drop_duplicates()

# Encoding (One-Hot Encoding untuk kolom 'cp')
# Di dataset kita, 'Chest Pain Type' bernama 'cp'
data = pd.get_dummies(data, columns=['cp'])

# Skalasi Fitur (Normalisasi Usia & Tekanan Darah)
scaler = StandardScaler()
data[['age', 'trestbps']] = scaler.fit_transform(data[['age', 'trestbps']])

# TAHAP 2: PEMROSESAN DATA (MEMBAGI DATA)
# Memisahkan fitur (X) dan target (y)
X = data.drop('target', axis=1)
y = data['target']

# Membagi data latih (80%) dan uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TAHAP 3: MELATIH DAN MENGUJI MODEL
# Memilih Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Melatih Model
model.fit(X_train, y_train)

# Evaluasi Model
y_pred = model.predict(X_test)

# Menampilkan Hasil Teks
akurasi = accuracy_score(y_test, y_pred)
print(f"\n=== HASIL EVALUASI ===")
print(f"Akurasi Model: {akurasi * 100:.2f}%\n")
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# TAHAP 4: VISUALISASI HASIL (BONUS NILAI PLUS)
# 1. Membuat Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Menggambar Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prediksi Sehat (0)', 'Prediksi Sakit (1)'],
            yticklabels=['Aktual Sehat (0)', 'Aktual Sakit (1)'])

plt.title('Confusion Matrix - Klasifikasi Penyakit Jantung')
plt.xlabel('Hasil Prediksi Model')
plt.ylabel('Kenyataan (Data Asli)')
plt.show()

# 2. Menampilkan Fitur Apa yang Paling Berpengaruh
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title('10 Fitur Paling Berpengaruh Terhadap Penyakit Jantung')
plt.xlabel('Tingkat Kepentingan (Importance)')
plt.show()