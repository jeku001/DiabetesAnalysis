import pandas as pd
import os

filePath_df_012 = os.path.join('..', 'data', 'diabetes_012_health_indicators_BRFSS2021.csv')
filePath_df_binary_5050 = os.path.join('..', 'data', 'diabetes_binary_5050split_health_indicators_BRFSS2021.csv')
filePath_df_binary = os.path.join('..', 'data', 'diabetes_binary_health_indicators_BRFSS2021.csv')

df_012 = pd.read_csv("diabetes_012_health_indicators_BRFSS2021.csv")

print(df_012.head())
