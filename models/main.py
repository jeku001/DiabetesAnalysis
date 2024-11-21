import os
import pandas as pd

#specified file path to import
filePath_df_012 = os.path.join('..', 'data', 'diabetes_012_health_indicators_BRFSS2021.csv')
filePath_df_binary_5050 = os.path.join('..', 'data', 'diabetes_binary_5050split_health_indicators_BRFSS2021.csv')
filePath_df_binary = os.path.join('..', 'data', 'diabetes_binary_health_indicators_BRFSS2021.csv')

#import data
df_012 = pd.read_csv(filePath_df_012)
df_binary_5050 = pd.read_csv(filePath_df_binary_5050)
df_binary = pd.read_csv(filePath_df_binary)

"""print(df_012.head())
print(df_binary_5050.head())
print(df_binary.head())"""

## wykres slupkowy liczebnosci
"""
labels = {0.0: 'No Diabetes', 1.0: 'Prediabetes', 2.0: 'Diabetes'}
counts_types_012 = df_012['Diabetes_012'].map(labels).value_counts().reindex(['No Diabetes', 'Prediabetes', 'Diabetes'])
plt.figure(figsize=(8, 5))
counts_types_012.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title("Group size")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=0)  # Opcjonalnie, aby wartości były wyświetlone poziomo
plt.show()
"""

## wykres slupkowy liczebnosci - csv binary
"""
labels = {0.0: 'No Diabetes', 1.0: 'Diabetes'}
counts_types_binary = df_binary['Diabetes_binary'].map(labels).value_counts().reindex(['No Diabetes', 'Diabetes'])
plt.figure(figsize=(8, 5))
counts_types_binary.plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title("Group size")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=0)  # Opcjonalnie, aby wartości były wyświetlone poziomo
plt.show()
"""


