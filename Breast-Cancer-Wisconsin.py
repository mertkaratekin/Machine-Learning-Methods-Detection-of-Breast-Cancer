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
from tkinter import *

# Veri Setini Okuma İşlemi
veri = pd.read_csv("breast-cancer-wisconsin.csv")

# Bağımlı Değişken Seçimi (Sınıflandırma için)
y_cls = veri["Class"]
x_cls = veri.drop(["Class"], axis=1)

# Veri Setinin Bölünmesi (Sınıflandırma için)
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
    x_cls, y_cls, train_size=0.6, random_state=46
)

# Verilerin Normalizasyonu (Sınıflandırma için)
scaler_cls = StandardScaler()
x_train_cls_scaled = scaler_cls.fit_transform(x_train_cls)
x_test_cls_scaled = scaler_cls.transform(x_test_cls)

# Sınıflandırma Modelleri
modeller = [
    KNeighborsClassifier(n_neighbors=9),
    MLPClassifier(hidden_layer_sizes=(200, 100), learning_rate_init=0.03, max_iter=10000, random_state=46),
    RandomForestClassifier(n_estimators=100, random_state=46),
    SVC(kernel='rbf', random_state=46),
    GaussianNB(),  # Naive Bayes
    DecisionTreeClassifier(random_state=46),  # Decision Trees
    LogisticRegression(random_state=46)  # Lojistik Regresyon
]

def sec_model(model):
    model.fit(x_train_cls_scaled, y_train_cls)
    tahminler = model.predict(x_test_cls_scaled)
    dogruluk = accuracy_score(y_test_cls, tahminler)
    hata_matrisi = confusion_matrix(y_test_cls, tahminler)
    siniflama_raporu = classification_report(y_test_cls, tahminler)
    
    # Grafik Oluşturma
    plt.figure(figsize=(10, 6))
    #plt.scatter(x_test_cls.iloc[:, 0], tahminler, c=y_test_cls, cmap='coolwarm', alpha=0.7)
    plt.scatter(x_test_cls.iloc[:, 2], tahminler, c=y_test_cls, cmap='coolwarm', alpha=0.7)
    plt.title(f"{model.__class__.__name__} Sınıflandırma Tahminleri (Doğruluk: {dogruluk:.4f})")
    plt.xlabel(x_test_cls.columns[8])
    plt.ylabel("Predicted class")
    plt.colorbar(label='Real Class')
    plt.show()
    
    # Performans Metrikleri Yazdırma
    print(f"\n{model.__class__.__name__} Sınıflandırma Performansı:\n")
    print(f"Doğruluk: {dogruluk:.4f}")
    print(f"Hata Matrisi:\n{hata_matrisi}")
    print(f"Sınıflama Raporu:\n{siniflama_raporu}")
    print("="*50)  # Ayırıcı çizgi

def kapat():
    root.destroy()

# Arayüz Oluşturma
root = Tk()
root.title("ALGORİTMA SEÇİM EKRANI")

frame = Frame(root)
frame.pack(padx=250, pady=150)

label_text = "ALGORİTMALAR"
label = Label(frame, text=label_text, font=("Arial", 14, "bold"), border=5)
label.grid(row=0, column=0, padx=50, pady=5)

def model_sec(index):
    selected_model = modeller[index]
    sec_model(selected_model)

for i, model in enumerate(modeller):
    btn = Button(frame, text=model.__class__.__name__, command=lambda index=i: model_sec(index), border= 3)
    btn.grid(row=i+1, column=0, padx=5, pady=5, sticky="ew")

kapat_btn = Button(frame, text="ÇIKIŞ", command=root.destroy, font=("Arial", 10, "bold"), background="red",  border= 3)
kapat_btn.grid(row=len(modeller)+1, column=0, padx=5, pady=5, sticky="ew")

root.mainloop()