import pickle

class Predict:
    def __init__(self, model_type, input_data):
        self.model_type = model_type
        self.input_data = input_data
        if self.model_type == 'survey':
            self.model_path = 'survey_model/RF_model.pkl'
        elif self.model_type == 'medical':
            self.model_path = 'medical_model/RF_model.pkl'
        else:
            raise ValueError("Model type must be 'survey' or 'medical'")

        self.model = self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the model: {e}")

    def predict(self):
        try:
            if len(self.input_data) != 15:
                raise ValueError("Input data must contain exactly 15 elements")
            prediction = self.model.predict([self.input_data])
            return int(prediction[0])
        except Exception as e:
            raise Exception(f"An error occurred during prediction: {e}")
