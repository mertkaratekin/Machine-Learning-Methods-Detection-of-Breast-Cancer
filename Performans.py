import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Veri Setini Okuma İşlemi
veri = pd.read_csv("Mikrokalsifikasyon.csv")

# Bağımlı Değişken Seçimi
y = veri["calification_type"]
veri.drop(["calification_type"], axis=1, inplace=True)
x = veri

# Veri Setinin Bölünmesi
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.4, random_state=46
)

# Normalizasyon
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Sınıflandırma Algoritmaları
algorithms = [
    ("Karar Ağacı", DecisionTreeClassifier(random_state=46)),
    ("Rastgele Orman", RandomForestClassifier(n_estimators=100, random_state=46)),
    ("Lojistik Regresyon", LogisticRegression(random_state=46)),
    ("Yapay Sinir Ağı", MLPClassifier(hidden_layer_sizes=(200, 100), learning_rate_init=0.03, max_iter=10000, random_state=46)),
    ("K-NN", KNeighborsClassifier(n_neighbors=9)),
    ("Destek Vektör Makineleri", SVC(kernel='rbf', random_state=46)),
    ("Naive Bayes", GaussianNB())
]

# Performans Metriklerini Hesapla
def calculate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Sınıflandırma ve Performans Hesaplama
performance_scores = {}
for name, algorithm in algorithms:
    algorithm.fit(x_train_scaled, y_train)
    y_pred = algorithm.predict(x_test_scaled)
    accuracy = calculate_performance(y_test, y_pred)
    performance_scores[name] = accuracy


# Sonuçları Grafikle Gösterme
plt.figure(figsize=(10, 6))
plt.bar(performance_scores.keys(), performance_scores.values(), color='blue')
plt.xlabel('Algoritmalar')
plt.ylabel('Doğruluk (Accuracy)')
plt.title('Sınıflandırma Algoritmalarının Performansı')
plt.ylim(0, 1)  # Y ekseni limiti 0 ile 1 arasında
plt.xticks(rotation=45, ha='right')
plt.show()
