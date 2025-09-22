import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import io

# --- 1. Persiapan Data ---

# Mencoba membaca file, jika gagal, gunakan data contoh
try:
    data = pd.read_csv('predic_tabel.csv')
except FileNotFoundError:
    print("File 'predic_tabel.csv' tidak ditemukan. Menggunakan data contoh.")
    file_content = """No,Usia,Jenis_Kelamin,Merokok,Bekerja,Rumah_Tangga,Aktivitas_Begadang,Aktivitas_Olahraga,Asuransi,Penyakit_Bawaan,Hasil
1,Tua,Pria,Pasif,Tidak,Ya,Ya,Sering,Ada,Tidak,Ya
2,Tua,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Ada,Tidak
3,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
4,Tua,Pria,Aktif,Ya,Tidak,Tidak,Jarang,Ada,Ada,Tidak
5,Muda,Wanita,Pasif,Ya,Tidak,Tidak,Sering,Tidak,Ada,Ya
6,Muda,Wanita,Pasif,Ya,Tidak,Tidak,Sering,Tidak,Ada,Tidak
7,Tua,Wanita,Pasif,Tidak,Ya,Tidak,Sering,Tidak,Tidak,Ya
8,Muda,Pria,Aktif,Tidak,Ya,Ya,Sering,Tidak,Tidak,Tidak
9,Tua,Wanita,Aktif,Ya,Ya,Ya,Jarang,Ada,Ada,Ya
10,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
"""
    data = pd.read_csv(io.StringIO(file_content))

# Menghapus kolom 'No' dan melakukan one-hot encoding
data = data.drop('No', axis=1)
data_encoded = pd.get_dummies(data, drop_first=True)
data_encoded.rename(columns={'Hasil_Ya': 'Hasil'}, inplace=True)

# Memisahkan fitur (X) dan target (y)
X = data_encoded.drop('Hasil', axis=1)
y = data_encoded['Hasil']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 2. Kelas untuk MODEL ---

class LinearRegression:
    """
    Mendefinisikan arsitektur model Regresi Linear.
    Tugasnya hanya menghitung prediksi dan gradien.
    """
    def __init__(self, n_features):  # Perbaikan: __init__
        # Inisialisasi bobot dan bias dengan nol
        self.weights = np.zeros(n_features)
        self.bias = 0

    def predict(self, X):
        # Menghitung prediksi: Y = (W * X) + b
        return np.dot(X, self.weights) + self.bias

    def get_gradients(self, X, y_true):
        # Menghitung seberapa besar kesalahan (gradien) untuk pembaruan
        y_predicted = self.predict(X)
        error = y_predicted - y_true
        
        # Turunan dari Mean Squared Error
        dw = (2 / X.shape[0]) * np.dot(X.T, error)
        db = (2 / X.shape[0]) * np.sum(error)
        
        return dw, db

# --- 3. Kelas untuk OPTIMIZER ---

class SGD_Optimizer:
    """
    Mendefinisikan algoritma optimisasi Stochastic Gradient Descent.
    Tugasnya adalah melatih sebuah model dengan memperbarui parameternya.
    """
    def __init__(self, model, learning_rate=0.01, n_iterations=1000):  # Perbaikan: __init__
        self.model = model
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples = X.shape[0]
        y = y.values  # Ubah ke numpy array

        # Loop training utama
        for _ in range(self.n_iterations):
            # Ambil satu sampel data secara acak untuk setiap iterasi
            random_index = np.random.randint(n_samples)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # 1. Minta model untuk menghitung gradiennya
            dw, db = self.model.get_gradients(xi, yi)

            # 2. Perbarui bobot dan bias milik model
            self.model.weights -= self.learning_rate * dw
            self.model.bias -= self.learning_rate * db


# --- 4. Menggabungkan dan Menjalankan Semuanya ---

# Tentukan jumlah fitur dari data latih
n_features = X_train_scaled.shape[1]

# 1. Buat instance dari model LinearRegression
linear_model = LinearRegression(n_features=n_features) # Perbaikan: Beri argumen n_features

# 2. Buat instance dari SGD_Optimizer dan berikan model di atas untuk dilatih
optimizer = SGD_Optimizer(model=linear_model, learning_rate=0.01, n_iterations=1000) # Perbaikan: Beri argumen yang diperlukan

# 3. Jalankan proses training
optimizer.fit(X_train_scaled, y_train)

# 4. Buat prediksi menggunakan model yang sudah dilatih
predictions_continuous = linear_model.predict(X_test_scaled)
predictions_class = [1 if val > 0.5 else 0 for val in predictions_continuous]


# --- 5. Evaluasi Hasil ---

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true.values - y_pred)**2)

def accuracy(y_true, y_pred):
    return np.sum(y_true.values == y_pred) / len(y_true)

print("--- Hasil Evaluasi Model ---")
print(f"Prediksi mentah (kontinu) dari model: {predictions_continuous}")
print(f"Prediksi setelah dikonversi ke kelas (0/1): {predictions_class}")
print(f"Nilai Sebenarnya: {y_test.values}")
print("\n--- Metrik Performa ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, predictions_continuous):.4f}")
print(f"Akurasi (setelah pembulatan): {accuracy(y_test, predictions_class):.2f}")