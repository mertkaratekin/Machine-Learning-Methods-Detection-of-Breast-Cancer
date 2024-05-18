import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Veri Setini Okuma İşlemi
#veri = pd.read_csv("Mikrokalsifikasyon.csv")
veri = pd.read_csv("BreastTissue.csv")

# Sayisal degerlerdeki noktalama işaretlerini kaldirma ve sayilari uygun formata dönuşturme
numeric_columns = ["I0", "PA500", "HFS", "DA", "Area", "A/DA", "Max IP", "DR", "P"]
veri[numeric_columns] = veri[numeric_columns].replace({'\.': '', ',': '.'}, regex=True).astype(float)

# Bagimli Degişken Seçimi (Siniflandirma için)
#y_cls = veri["calification_type"]
#x_cls = veri.drop(["calification_type"], axis=1)
y_cls = veri["Class"]
x_cls = veri.drop(["Class"], axis=1)

# Veri Setinin Bölunmesi (Siniflandirma için)
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
    x_cls, y_cls, train_size=0.7, random_state=46
)

# Verilerin Normalizasyonu (Siniflandirma için)
scaler_cls = StandardScaler()
x_train_cls_scaled = scaler_cls.fit_transform(x_train_cls)
x_test_cls_scaled = scaler_cls.transform(x_test_cls)

# Siniflandirma Modelleri
modeller = [
    KNeighborsClassifier(n_neighbors=9),
    MLPClassifier(hidden_layer_sizes=(200, 100), learning_rate_init=0.03, max_iter=10000, random_state=46),
    RandomForestClassifier(n_estimators=100, random_state=46),
    SVC(kernel='rbf', random_state=46),
    GaussianNB(),  # Naive Bayes
    DecisionTreeClassifier(random_state=46),  # Decision Trees
    LogisticRegression(random_state=46)  # Lojistik Regresyon
]

# Siniflandirma Tahminleri ve Performans Degerleri
for i, model in enumerate(modeller):
    model.fit(x_train_cls_scaled, y_train_cls)
    tahminler = model.predict(x_test_cls_scaled)
    dogruluk = accuracy_score(y_test_cls, tahminler)
    hata_matrisi = confusion_matrix(y_test_cls, tahminler)
    siniflama_raporu = classification_report(y_test_cls, tahminler)
    
    # Grafik Oluşturma
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test_cls.iloc[:, 0], tahminler, c=y_test_cls, cmap='coolwarm', alpha=0.7)
    plt.title(f"{model.__class__.__name__} Siniflandirma Tahminleri (Dogruluk: {dogruluk:.4f})")
    plt.xlabel(x_test_cls.columns[0])
    plt.ylabel("Predicted class")
    plt.colorbar(label='Real Class')
    plt.show()
    
    # Performans Metrikleri Yazdirma
    print(f"\n{model.__class__.__name__} Siniflandirma Performansi:\n")
    print(f"Dogruluk: {dogruluk:.4f}")
    print(f"Hata Matrisi:\n{hata_matrisi}")
    print(f"Siniflama Raporu:\n{siniflama_raporu}")
    print("="*50)  # Ayirici çizgi
