import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, classification_report
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def analyse_data(data):
    num_data = len(data)
    print(f"Toplam Kayıt Sayısı: {num_data}\n")

    print("Nitelik isimleri ve her sütuna ait olan eşşiz değer sayıları: ")
    for column in data.columns:
        num_attribute = data[column].nunique()
        most_freq_value = data[column].mode()[0]
        print(
            f"{column}: {num_attribute} adet değer var, en çok tekrar eden veri: {most_freq_value}"
        )

    print(f"Sütun Nitelik Tipleri:\n{data.dtypes}")


def categorical_statistics(data):
    print("İşlenmiş verinin görselleştirilmiş hali: ")
    categorical_colums = data.select_dtypes(include="object").columns
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(categorical_colums, 1):
        plt.subplot(4, 5, i)
        data[column].value_counts().plot(kind="bar", color="skyblue")
        plt.title(f"{column} Sütun Grafiği")
        plt.xlabel(column)
        plt.ylabel("Frekans")
    plt.tight_layout()
    plt.show()


def preprocessing(adress):
    data = pd.read_excel(adress)
    print("İşlenmemiş verinin boyutu: ", len(data))
    categorical_statistics(data)
    data = data.drop_duplicates()
    print("Verideki tekrar eden değerlerin silinmesinden sonraki boyut: ", len(data))
    data["Cihaz Ağırlığı"] = data["Cihaz Ağırlığı"].replace("1 - 2 kg", "2 kg ve Altı")
    data["İşletim Sistemi"] = data["İşletim Sistemi"].apply(
        lambda x: "Windows" if x.startswith("W") else x
    )
    data["İşlemci"] = (
        data["İşlemci"]
        .astype(str)
        .apply(
            lambda x: (
                "Intel"
                if x.endswith(
                    ("H", "U", "J", "K", "F", "P", "G1", "G7", "G4", "HX", "HS", "G16")
                )
                or x.startswith("10.")
                else "AMD" if x.startswith(("AMD", "Ryzen")) else "Diğer"
            )
        )
    )
    data["Max Ekran Çözünürlüğü"] = data["Max Ekran Çözünürlüğü"].apply(
        lambda x: x if x == "1920 x 1080" else "Diğer"
    )

    # En çok veriye sahip olan 5 değeri bul
    top_5_values = data["Ekran Boyutu"].value_counts().nlargest(5).index

    # "Diğer" olarak işaretlemek için koşul belirle
    condition = ~data["Ekran Boyutu"].isin(top_5_values)

    # Koşulu sağlayanları "Diğer" olarak güncelle
    data.loc[condition, "Ekran Boyutu"] = "Diğer"

    # En çok veriye sahip olan 5 değeri bul
    top_4_values = data["Bellek Hızı"].value_counts().nlargest(4).index

    # "Diğer" olarak işaretlemek için koşul belirle
    condition = ~data["Bellek Hızı"].isin(top_4_values)

    # Koşulu sağlayanları "Diğer" olarak güncelle
    data.loc[condition, "Bellek Hızı"] = "Diğer"

    # En çok veriye sahip olan 5 değeri bul
    top_7_values = data["Maksimum İşlemci Hızı"].value_counts().nlargest(7).index

    # "Diğer" olarak işaretlemek için koşul belirle
    condition = ~data["Maksimum İşlemci Hızı"].isin(top_7_values)

    # Koşulu sağlayanları "Diğer" olarak güncelle
    data.loc[condition, "Maksimum İşlemci Hızı"] = "Diğer"

    # Kategorik değerleri sayısal formata dönüştürün
    data = data.replace("", pd.NA)
    categorical_statistics(data)
    X = data.drop("Fiyat", axis=1)
    y = data["Fiyat"]
    y_encoded = pd.get_dummies(y)
    data_encoded = pd.get_dummies(X)
    # Eksik değerleri doldurun
    imputer = SimpleImputer(strategy="most_frequent")
    data_filled = pd.DataFrame(
        imputer.fit_transform(data_encoded), columns=data_encoded.columns
    )
    save_to_excel(data_filled, "islenmisdata.xlsx")
    return data_filled, y_encoded


def save_to_excel(data_filled, output_excel):
    data_filled.to_excel(output_excel)


def classification(data_filled, y_encoded):
    # Eğitim ve test setlerini oluşturun
    X_train, X_test, y_train, y_test = train_test_split(
        data_filled, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100)

    # Modeli eğitin
    model.fit(data_filled, y_encoded)

    with open("C:\\Users\\90552\\Desktop\\model.pkl", "wb") as file:
        pickle.dump(model, file)

    return model


def test(model, test_data, y_encoded):
    # Model üzerinde tahmin yapın
    y_pred = model.predict(test_data)
    # Sınıflandırma raporunu gösterin
    classification_rep = classification_report(
        y_encoded.values.argmax(axis=1), y_pred.argmax(axis=1)
    )
    print("Sınıflandırma Raporu:\n", classification_rep)


if __name__ == "__main__":

    adress = "C:/Users/90552/Desktop/data.xlsx"
    analyse_data(pd.read_excel(adress))
    test_adress = "C:/Users/90552/Desktop/test.xlsx"
    # Eğitim ve modeli alma

    data_filled, y_encoded = preprocessing(adress)

    trained_model = classification(data_filled, y_encoded)
    test_filled, y_encoded = preprocessing(test_adress)

    farklideger = set(test_filled.columns) - set(data_filled.columns)
    test_filled = test_filled.drop(columns=farklideger)

    for column in data_filled.columns:
        if column not in test_filled.columns:
            test_filled[column] = 0

    test_filled = test_filled[data_filled.columns]

    with open("C:\\Users\\90552\\Desktop\\model.pkl", "rb") as file:
        model = pickle.load(file)
    test(model, test_filled, y_encoded)
