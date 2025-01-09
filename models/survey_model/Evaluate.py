from sklearn.metrics import accuracy_score, classification_report

class Evaluate:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)