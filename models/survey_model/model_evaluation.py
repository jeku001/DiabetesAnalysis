import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

class ModelEvaluation:
    def __init__(self, file_path):
        self.data = pd.read_pickle(file_path)
        self.X = self.data.drop("Diabetes_binary", axis=1)
        self.y = self.data["Diabetes_binary"]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_and_save_models(self):
        # Logistic Regression
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(self.X_train, self.y_train)

        # Save Logistic Regression model
        with open("logistic_model.pkl", "wb") as file:
            pickle.dump(logistic_model, file)

        print("Logistic Regression model saved as logistic_model.pkl")

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)

        # Save Random Forest model
        with open("random_forest_model.pkl", "wb") as file:
            pickle.dump(rf_model, file)

        print("Random Forest model saved as random_forest_model.pkl")

    def evaluate_models(self):
        # Logistic Regression Evaluation
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(self.X_train, self.y_train)
        logistic_predictions = logistic_model.predict(self.X_test)
        print("Logistic Regression Evaluation:")
        print(f"Accuracy: {accuracy_score(self.y_test, logistic_predictions):.2f}")
        print(classification_report(self.y_test, logistic_predictions))

        # Random Forest Evaluation
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_predictions = rf_model.predict(self.X_test)
        print("Random Forest Evaluation:")
        print(f"Accuracy: {accuracy_score(self.y_test, rf_predictions):.2f}")
        print(classification_report(self.y_test, rf_predictions))

if __name__ == "__main__":
    evaluator = ModelEvaluation("df_binary_5050_reduced.pkl")
    evaluator.split_data()
    evaluator.train_and_save_models()
    evaluator.evaluate_models()
