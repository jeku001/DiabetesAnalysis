import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import f1_score

# Wczytanie danych
train_data = pd.read_csv("train/in.tsv", sep="\t", header=None)
train_labels = pd.read_csv("train/expected.tsv", sep="\t", header=None)
test_data = pd.read_csv("test/in.tsv", sep="\t", header=None)

# Zduplikowanie mniejszej klasy
minority_class = train_data[train_labels[0] == 1]
majority_class = train_data[train_labels[0] == 0]

minority_class_resampled = resample(minority_class,
                                    replace=True,
                                    n_samples=len(majority_class),
                                    random_state=42)

# Połączenie zduplikowanej mniejszości z większością
balanced_train_data = pd.concat([majority_class, minority_class_resampled])
balanced_train_labels = pd.concat([train_labels[train_labels[0] == 0],
                                    train_labels.loc[minority_class_resampled.index]])

# Dopasowanie modelu
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(balanced_train_data, balanced_train_labels.values.ravel())

# Predykcje na danych testowych
test_predictions = model.predict(test_data)

# Zapisanie wyników do pliku
pd.DataFrame(test_predictions).to_csv("test/expected.tsv", sep="\t", index=False, header=False)

# Obliczenie F1-score
# Wczytanie rzeczywistych etykiet testowych
test_actual_labels = pd.read_csv("expected_real.tsv", sep="\t", header=None)
f1 = f1_score(test_actual_labels, test_predictions)

print(f"F1 Score: {f1}")
