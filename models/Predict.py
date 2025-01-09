import pickle

class Predict:
    def __init__(self, model_type, input_data):
        self.model_type = model_type
        self.input_data = input_data
        print(self.input_data)  # debug
        if self.model_type == 'survey':
            self.model_path = 'survey_model/RF_model.pkl'
            self.expected_input_length = 15
        elif self.model_type == 'medical':
            self.model_path = 'medical_model/RF_model_medical.pkl'
            self.expected_input_length = 8
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
            if len(self.input_data) != self.expected_input_length:
                raise ValueError(f"Input data must contain exactly {self.expected_input_length} elements")
            prediction = self.model.predict([self.input_data])
            return int(prediction[0])
        except Exception as e:
            raise Exception(f"An error occurred during prediction: {e}")
