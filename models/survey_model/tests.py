import pandas as pd
from Train import Train
from Evaluate import Evaluate
import pickle
from sklearn.metrics import accuracy_score, classification_report

"""data = pd.read_pickle("df_binary_reduced.pkl")
test_data_removed_90percent = data.sample(frac=0.1, random_state=42)  # Zachowujemy 10% danych
test_data_removed_90percent.to_pickle("test_data_removed_90percent.pkl")
#print(data) # 236378 rows
#print(data.columns)"""

"""# Wczytanie danych z plików pickle
processed_data = pd.read_pickle("processed_data.pkl")
df_binary_reduced = pd.read_pickle("df_binary_reduced.pkl")

# Sprawdzenie rozkładu liczności dla kolumny 'Diabetes_binary'
print("Rozkład dla 'Diabetes_binary' w processed_data:")
print(processed_data['Diabetes_binary'].value_counts())

print("\nRozkład dla 'Diabetes_binary' w df_binary_reduced:")
print(df_binary_reduced['Diabetes_binary'].value_counts())

# Liczenie wierszy z co najmniej jednym NaN
nan_count_processed_data = processed_data.isna().any(axis=1).sum()
nan_count_df_binary_reduced = df_binary_reduced.isna().any(axis=1).sum()

print("\nLiczba wierszy z co najmniej jednym NaN w processed_data:", nan_count_processed_data)
print("Liczba wierszy z co najmniej jednym NaN w df_binary_reduced:", nan_count_df_binary_reduced)
"""
"""

Rozkład dla 'Diabetes_binary' w processed_data:
Diabetes_binary
1.0    304538
0.0    303892
Name: count, dtype: int64

Rozkład dla 'Diabetes_binary' w df_binary_reduced:
Diabetes_binary
0.0    202810
1.0     33568
Name: count, dtype: int64

Liczba wierszy z co najmniej jednym NaN w processed_data: 202810
Liczba wierszy z co najmniej jednym NaN w df_binary_reduced: 0

Process finished with exit code 0


"""
#---------------------------------------------------------#
# TRENOWANIE I ZAPISANIE MODELU RF


"""

# Przykład użycia
data = pd.read_pickle("processed_data.pkl")
train_instance = Train(data, target_column='Diabetes_binary')
train_instance.train()

evaluate_instance = Evaluate(train_instance.model, train_instance.X_test, train_instance.y_test)
evaluate_instance.evaluate()

"""


"""

Model trained successfully.
Model saved to RF_model.pkl
Accuracy: 0.8764607267886199
Classification Report:
              precision    recall  f1-score   support

         0.0       0.86      0.90      0.88     60926
         1.0       0.89      0.85      0.87     60760

    accuracy                           0.88    121686
   macro avg       0.88      0.88      0.88    121686
weighted avg       0.88      0.88      0.88    121686


Process finished with exit code 0


"""
"""
# Ścieżka do modelu i danych
model_path = 'RF_model.pkl'
data_path = 'df_binary_reduced.pkl'
target_column = 'Diabetes_binary'

# Ładowanie modelu
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Ładowanie danych
def load_data(data_path, target_column):
    data = pd.read_pickle(data_path)
    X_test = data.drop(columns=[target_column])
    y_test = data[target_column]
    return X_test, y_test

# Ocena modelu
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

# Ładowanie modelu i danych
model = load_model(model_path)
X_test, y_test = load_data(data_path, target_column)

# Ocena modelu
evaluate_model(model, X_test, y_test)
"""

"""

Accuracy: 0.9487177317686079
Classification Report:
              precision    recall  f1-score   support

         0.0       0.97      0.97      0.97    202810
         1.0       0.81      0.83      0.82     33568

    accuracy                           0.95    236378
   macro avg       0.89      0.90      0.90    236378
weighted avg       0.95      0.95      0.95    236378



"""

df = pd.read_pickle("processed_data.pkl")
print(df.columns)

"""

Index(['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump',
       'GenHlth', 'MentHlth', 'DiffWalk', 'Sex', 'Age', 'Income'],
      dtype='object')

"""