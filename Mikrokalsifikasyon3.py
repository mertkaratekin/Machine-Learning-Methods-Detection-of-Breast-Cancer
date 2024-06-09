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

# Fonksiyon: Performans Degerlerini Hesapla
def performance_calculate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, confusion, report

# Veri Setini Okuma İşlemi
veri = pd.read_csv("Mikrokalsifikasyon.csv")

# Bağimli Değişken Seçimi
y_cls = veri["calification_type"]
veri.drop(["calification_type"], axis=1, inplace=True)
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

# Performans Ölçütleri
performanslar = []
for model in modeller:
    model.fit(x_train_cls_scaled, y_train_cls)
    tahminler = model.predict(x_test_cls_scaled)
    accuracy, confusion, report = performance_calculate(y_test_cls, tahminler)
    performanslar.append((accuracy, confusion, report))

# Performans Tablosu
df_cls = pd.DataFrame(performanslar, index=[model.__class__.__name__ for model in modeller],
                      columns=["Dogruluk", "Hata Matrisi", "Siniflama Raporu"])

print("\n\nSiniflandirma Performans Tablosu:\n")
print(df_cls.to_string())

# Tahminlerin Gösterilmesi ve Grafiklerin Oluşturulması
for i, model in enumerate(modeller):
    tahminler = model.predict(x_test_cls_scaled)
    
    # Tahminlerin Gösterilmesi
    print(f"\n{model.__class__.__name__} Predictions and Real Values:")
    tahmin_df = pd.DataFrame({"Real Values": y_test_cls, "Prediction": tahminler})
    print(tahmin_df.to_string())
    
    # Doğruluk Yüzdesi Hesaplanması
    accuracy = accuracy_score(y_test_cls, tahminler)
    accuracy_percentage = round(accuracy * 100, 2)
    print(f"{model.__class__.__name__} doğruluk yüzdesi: {accuracy_percentage}%")
    
    # Grafik Oluşturma
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test_cls.iloc[:, 0], tahminler, c=y_test_cls, cmap='coolwarm', alpha=0.7)
    plt.title(f"{model.__class__.__name__} Classification Predictions (Accuracy: {accuracy_percentage}%)")
    plt.xlabel(x_cls.columns[0])  # İstediğiniz sütunu x eksenine yerleştirin
    plt.ylabel("Forecast Values")
    plt.colorbar(label='Real Class')
    plt.show()


    
