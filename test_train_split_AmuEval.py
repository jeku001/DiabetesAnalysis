import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytaj dane z pliku CSV
data = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2021.csv")

# Podział na etykiety (pierwsza kolumna) i cechy (pozostałe kolumny)
labels = data.iloc[:, 0]
features = data.iloc[:, 1:]

# Podział na zbiór treningowy i testowy (70% - trening, 30% - test)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Dodanie nagłówków kolumn do plików
feature_headers = features.columns.tolist()
label_header = [data.columns[0]]

# Zapisz dane do plików TSV z nagłówkami
features_train.to_csv("in.tsv", sep="\t", index=False, header=True)
labels_train.to_frame().to_csv("expected.tsv", sep="\t", index=False, header=label_header)
features_test.to_csv("in.tsv", sep="\t", index=False, header=True)
labels_test.to_frame().to_csv("expected.tsv", sep="\t", index=False, header=label_header)

print("Pliki zostały utworzone z nagłówkami.")
