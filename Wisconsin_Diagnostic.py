import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tkinter import *

# Fonksiyon: Performans Degerlerini Hesapla
def performance_calculate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, confusion, report

# Veri Setini Okuma İşlemi
veri = pd.read_csv("Wisconsin_Diagnostic.csv")

# Bağimli Değişken Seçimi
y_cls = veri["Diagnosis"]
veri.drop(["Diagnosis"], axis=1, inplace=True)
x_cls = veri

# Veri Setinin Bölünmesi
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
    x_cls, y_cls, train_size=0.4, random_state=46
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

def perform_classification(selected_models):
    performanslar = []
    for model_idx in selected_models:
        model = modeller[model_idx]
        model.fit(x_train_cls_scaled, y_train_cls)
        tahminler = model.predict(x_test_cls_scaled)
        accuracy, confusion, report = performance_calculate(y_test_cls, tahminler)
        performanslar.append((accuracy, confusion, report))

        # Tahminlerin Gösterilmesi
        print(f"\n{model.__class__.__name__} Tahminleri ve Gerçek Değerler:")
        tahmin_df = pd.DataFrame({"Gerçek Değer": y_test_cls, "Tahmin": tahminler})
        print(tahmin_df.to_string())

        # Grafik Oluşturma
        plt.figure(figsize=(10, 6))
        plt.scatter(x_test_cls.iloc[:, 0], tahminler, c=y_test_cls, cmap='coolwarm', alpha=0.7)
        plt.title(f"{model.__class__.__name__} Sınıflandırma Tahminleri (Doğruluk: {accuracy:.4f})", fontsize=16)
        plt.xlabel(x_cls.columns[7], fontsize=14)  # İstediğiniz sütunu x eksenine yerleştirin
        plt.ylabel("Tahmin Edilen Değerler", fontsize=14)
        plt.colorbar(label='Real Class')
        
        # Tahmin edilen ve gerçek değerlerin altında raporun yazdırılması
        plt.text(0, -1, f"\n\n\n\n\nSınıflandırma Raporu:\n{report}", fontsize=12, ha='left', va='top')
        
        plt.show()

# Arayüz Fonksiyonları
def toggle_model(idx):
    if idx in selected_models:
        selected_models.remove(idx)
    else:
        selected_models.append(idx)

def classify():
    perform_classification(selected_models)

def close_program():
    root.destroy()



# Arayüz Oluşturma
root = Tk()
root.title("Sınıflandırma Arayüzü")

# Pencere boyutunu ve yazı tipi boyutunu ayarla
root.geometry("800x700")
root.config(bg="#1E1E1E")  # Arka plan rengini değiştirdik

# Başlık alanı oluştur
title_label = Label(root, text="ALGORİTMALAR", font=("Arial", 24, "bold"), bg="#1E1E1E", fg="#04F943")  # Başlığın arka plan rengini ayarladık
title_label.pack(pady=20)

# Çerçeve oluştur
frame = Frame(root, bg="#1E1E1E", bd=5)  # Çerçevenin arka plan rengini ayarladık ve kenar kalınlığını arttırdık
frame.pack(pady=20, padx=20)  # Boşlukları arttırdık

selected_models = []
#color_list = ["#CCFFCC","#99CC99","#669966","#99FF99","#66CC66","#336633","#339933"]
color_list = ["#FF0000","#FF7F00","#FFFF00","#00FF00","#0000FF","#4B0082","#9400D3"] 
j = 0

for i, model in enumerate(modeller):
    btn = Checkbutton(frame, text=model.__class__.__name__, command=lambda idx=i: toggle_model(idx), font=("Arial", 14, "bold"), bg = color_list[j], fg="#1E1E1E", bd=0, padx=20, pady=10, selectcolor="#64b5f6")
    btn.grid(row=i, column=0, sticky="w", pady=5)  # Butonlar arasında biraz boşluk bıraktık
    j = j + 1 

classify_button = Button(root, text="SINIFLANDIR", command=classify, font=("Arial", 16, "bold"), bg="#04F943", fg="black", bd=0, padx=20, pady=10, relief="flat")
classify_button.pack(pady=20)

close_button = Button(root, text="ÇIKIŞ", command=close_program, font=("Arial", 16, "bold"),bg = "#FF0000", fg="black", bd=0, padx=20, pady=10, relief="flat")
close_button.pack()

root.mainloop()
