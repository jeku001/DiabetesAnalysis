from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pickle

class TrainAndEvaluate:
    def __init__(self, data, target_column, test_size=0.2, random_state=None):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()

    def _prepare_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        self.evaluate()
        self.save_model()

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions, average='macro')
        f1 = f1_score(self.y_test, predictions, average='macro')
        print(f"Evaluation Results: Accuracy = {accuracy:.2f}, Recall = {recall:.2f}, F1-Score = {f1:.2f}")

    def save_model(self):
        with open('RF_model_medical.pkl', 'wb') as file:
            pickle.dump(self.model, file)


