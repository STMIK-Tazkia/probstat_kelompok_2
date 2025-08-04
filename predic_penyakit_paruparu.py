
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# data predic penyakit paru paru
csv_data = """No,Usia,Jenis_Kelamin,Merokok,Bekerja,Rumah_Tangga,Aktivitas_Begadang,Aktivitas_Olahraga,Asuransi,Penyakit_Bawaan,Hasil
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
11,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
12,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
13,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
14,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
15,Muda,Wanita,Pasif,Ya,Tidak,Ya,Sering,Tidak,Ada,Ya
16,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
17,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
18,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
19,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
20,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
21,Muda,Wanita,Pasif,Ya,Tidak,Ya,Sering,Tidak,Ada,Ya
22,Tua,Pria,Pasif,Tidak,Ya,Ya,Sering,Ada,Tidak,Ya
23,Tua,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Ada,Tidak
24,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
25,Tua,Pria,Aktif,Ya,Tidak,Tidak,Jarang,Ada,Ada,Tidak
26,Muda,Wanita,Pasif,Ya,Tidak,Tidak,Sering,Tidak,Ada,Ya
27,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
28,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
29,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
30,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
31,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
32,Muda,Wanita,Pasif,Ya,Tidak,Tidak,Sering,Tidak,Ada,Tidak
33,Tua,Wanita,Pasif,Tidak,Ya,Tidak,Sering,Tidak,Tidak,Ya
34,Muda,Pria,Aktif,Tidak,Ya,Ya,Sering,Tidak,Tidak,Tidak
35,Tua,Wanita,Aktif,Ya,Ya,Ya,Jarang,Ada,Ada,Ya
36,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
37,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
38,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
39,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
40,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
41,Muda,Wanita,Pasif,Ya,Tidak,Ya,Sering,Tidak,Ada,Ya
42,Tua,Pria,Aktif,Ya,Tidak,Tidak,Jarang,Ada,Ada,Tidak
43,Muda,Wanita,Pasif,Ya,Tidak,Tidak,Sering,Tidak,Ada,Ya
44,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
45,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
46,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
47,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
48,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
49,Muda,Wanita,Pasif,Ya,Tidak,Tidak,Sering,Tidak,Ada,Tidak
50,Tua,Wanita,Pasif,Tidak,Ya,Tidak,Sering,Tidak,Tidak,Ya
51,Muda,Pria,Aktif,Tidak,Ya,Ya,Sering,Tidak,Tidak,Tidak
52,Tua,Wanita,Aktif,Ya,Ya,Ya,Jarang,Ada,Ada,Ya
53,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
54,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
55,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
56,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
57,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
58,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
59,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
60,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
61,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
62,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
63,Muda,Wanita,Pasif,Ya,Tidak,Ya,Sering,Tidak,Ada,Ya
64,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
65,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
66,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
67,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
68,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
69,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
70,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
71,Muda,Wanita,Pasif,Ya,Tidak,Tidak,Sering,Tidak,Ada,Tidak
72,Tua,Wanita,Pasif,Tidak,Ya,Tidak,Sering,Tidak,Tidak,Ya
73,Muda,Pria,Aktif,Tidak,Ya,Ya,Sering,Tidak,Tidak,Tidak
74,Tua,Wanita,Aktif,Ya,Ya,Ya,Jarang,Ada,Ada,Ya
75,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
76,Muda,Pria,Aktif,Tidak,Ya,Ya,Jarang,Ada,Tidak,Tidak
77,Tua,Wanita,Aktif,Ya,Tidak,Ya,Jarang,Ada,Ada,Tidak
78,Muda,Wanita,Pasif,Ya,Tidak,Ya,Jarang,Ada,Ada,Ya
79,Tua,Wanita,Pasif,Ya,Ya,Tidak,Sering,Ada,Ada,Ya
80,Tua,Wanita,Aktif,Tidak,Ya,Tidak,Jarang,Ada,Tidak,Tidak
"""


data = pd.read_csv(StringIO(csv_data))
data = data.head(60)

data = data.drop('No', axis=1)

categorical_features = data.columns.drop('Hasil')

encoder = OrdinalEncoder()

data[categorical_features] = encoder.fit_transform(data[categorical_features])

X = data[categorical_features]
y = data['Hasil']
.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CategoricalNB(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"Jumlah baris data yang digunakan: {len(data)}")
print(f"Ukuran data latih: {len(X_train)} baris")
print(f"Ukuran data uji: {len(X_test)} baris")
print(f"\nAkurasi Model Naive Bayes : {accuracy:.2f}")
print("\nLaporan Klasifikasi:")
print(report)
print("\nMatriks Konfusi:")

print(cm)

