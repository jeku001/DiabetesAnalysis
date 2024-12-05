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
X = data.drop(columns=["Diabetes_binary"])  # Drop the target variable from features
y = data["Diabetes_binary"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize linear regression model
linearreg = LinearRegression()

# Sequential feature selector for forward selection
forwad = SequentialFeatureSelector(
    linearreg,
    k_features=15,  # Number of features to keep
    forward=False,
    verbose=1,
    scoring="neg_mean_squared_error"
)

# Fit the selector
sf = forwad.fit(X_train, y_train)

# Get selected feature names
selected_feat_names = list(sf.k_feature_names_)
print("Selected features:", selected_feat_names)

# Get all feature names
all_feat_names = X.columns.tolist()

# Calculate removed features
removed_feat_names = list(set(all_feat_names) - set(selected_feat_names))
print("Removed features:", removed_feat_names)

# Add a constant to the model (intercept) for logistic regression
X_train_const = sm.add_constant(X_train)

# Fit the logistic regression model on the training data
model = sm.Logit(y_train, X_train_const).fit(disp=0)  # Suppress verbose output

# Get p-values and coefficients
p_values = model.pvalues
coefficients = model.params

# Create a DataFrame to display feature names, coefficients, and p-values
results = pd.DataFrame({
    'Feature': p_values.index,
    'Coefficient': coefficients,
    'P-Value': p_values
})

# Sort the DataFrame by P-Value for better readability
results_sorted = results.sort_values(by='P-Value')

# Output the results
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

