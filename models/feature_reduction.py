import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
import statsmodels.api as sm


#specified file path to import
filePath_df_012 = os.path.join('..', 'data', 'diabetes_012_health_indicators_BRFSS2021.csv')
filePath_df_binary_5050 = os.path.join('..', 'data', 'diabetes_binary_5050split_health_indicators_BRFSS2021.csv')
filePath_df_binary = os.path.join('..', 'data', 'diabetes_binary_health_indicators_BRFSS2021.csv')

#import data
df_012 = pd.read_csv(filePath_df_012)
df_binary_5050 = pd.read_csv(filePath_df_binary_5050)
df_binary = pd.read_csv(filePath_df_binary)


## regresyjna wsteczna eliminacja pozbede sie niepotrzebnych cech

data = df_binary
X = data.drop(columns=["Diabetes_binary"])
y = data["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


linearreg = LinearRegression()
backward = SequentialFeatureSelector(
    linearreg,
    k_features=15,  # Number of features to keep
    forward=False,
    verbose=1,
    scoring="neg_mean_squared_error"
)

sf = backward.fit(X_train, y_train)

selected_feat_names = list(sf.k_feature_names_)
print("Selected features:", selected_feat_names)
all_feat_names = X.columns.tolist()
removed_feat_names = list(set(all_feat_names) - set(selected_feat_names))
print("Removed features:", removed_feat_names)
X_train_const = sm.add_constant(X_train)
model = sm.Logit(y_train, X_train_const).fit(disp=0)  # Suppress verbose output
p_values = model.pvalues
coefficients = model.params

results = pd.DataFrame({
    'Feature': p_values.index,
    'Coefficient': coefficients,
    'P-Value': p_values
})

results_sorted = results.sort_values(by='P-Value')

print("Feature influence on diabetes (sorted by p-value):")
print(results_sorted)

## zostawiam 15 najbardziej znaczacych cech

## tutaj zapisuje do pickle files bo wygodnie
df_012_reduced = df_012.drop(columns=removed_feat_names)
df_binary_reduced = df_binary.drop(columns=removed_feat_names)
df_binary_5050_reduced = df_binary_5050.drop(columns=removed_feat_names)
df_012_reduced.to_pickle('df_012_reduced.pkl')
df_binary_reduced.to_pickle('df_binary_reduced.pkl')
df_binary_5050_reduced.to_pickle('df_binary_5050_reduced.pkl')

