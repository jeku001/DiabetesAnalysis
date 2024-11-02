## regresja regresyjna wsteczna eliminacja pozbede sie niepotrzebnych cech

# Define the target variable and features
data = df_binary
X = data.drop(columns=["Diabetes_binary"])  # Drop the target variable from features
y = data["Diabetes_binary"]  # Target variable

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

linearreg = LinearRegression() ## wybrana regresja
forwad = SequentialFeatureSelector(
linearreg,
k_features=15, ## ile zostawiamy cech
forward=True,
verbose=1,
scoring="neg_mean_squared_error"
)
sf = forwad.fit(X,y)

remaining_feat_names = list(sf.k_feature_names_)
print(remaining_feat_names)
all_feat_names = X.columns.tolist()
removed_feat_names = list(set(all_feat_names) - set(remaining_feat_names))
print("Removed features:", removed_feat_names)
print(sf.k_feature_idx_)

## remaining features
##['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'DiffWalk', 'Sex', 'Age', 'Income']
##(0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 16, 17, 18, 20)

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
