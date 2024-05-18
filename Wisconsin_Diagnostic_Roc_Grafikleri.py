import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tkinter import *

# Fonksiyon: Performans Değerlerini Hesapla
def performance_calculate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, confusion, report

# Veri Setini Okuma İşlemi
veri = pd.read_csv("Wisconsin_Diagnostic.csv")

# Bağımlı Değişken Seçimi
y_cls = veri["Diagnosis"]
veri.drop(["Diagnosis"], axis=1, inplace=True)
x_cls = veri

# Veri Setinin Bölünmesi
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
    x_cls, y_cls, train_size=0.4, random_state=46
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
    SVC(kernel='rbf', random_state=46, probability=True),
    GaussianNB(),  # Naive Bayes
    DecisionTreeClassifier(random_state=46),  # Decision Trees
    LogisticRegression(random_state=46)  # Lojistik Regresyon
]

def perform_classification(selected_models):
    plt.figure(figsize=(12, 8))
    for model_idx in selected_models:
        model = modeller[model_idx]
        model.fit(x_train_cls_scaled, y_train_cls)
        y_prob = model.predict_proba(x_test_cls_scaled)[:, 1]  # Positive class probabilities
        fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
        auc_score = roc_auc_score(y_test_cls, y_prob)
        plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right')
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

root.geometry("800x700")
root.config(bg="#1E1E1E")

title_label = Label(root, text="ALGORİTMALAR", font=("Arial", 24, "bold"), bg="#1E1E1E", fg="#04F943")
title_label.pack(pady=20)

frame = Frame(root, bg="#1E1E1E", bd=5)
frame.pack(pady=20, padx=20)

selected_models = []
color_list = ["#FF0000","#FF7F00","#FFFF00","#00FF00","#0000FF","#4B0082","#9400D3"] 
j = 0

for i, model in enumerate(modeller):
    btn = Checkbutton(frame, text=model.__class__.__name__, command=lambda idx=i: toggle_model(idx), font=("Arial", 14, "bold"), bg = color_list[j], fg="#1E1E1E", bd=0, padx=20, pady=10, selectcolor="#64b5f6")
    btn.grid(row=i, column=0, sticky="w", pady=5)
    j = j + 1 

classify_button = Button(root, text="SINIFLANDIR", command=classify, font=("Arial", 16, "bold"), bg="#04F943", fg="black", bd=0, padx=20, pady=10, relief="flat")
classify_button.pack(pady=20)

close_button = Button(root, text="ÇIKIŞ", command=close_program, font=("Arial", 16, "bold"),bg = "#FF0000", fg="black", bd=0, padx=20, pady=10, relief="flat")
close_button.pack()

root.mainloop()
