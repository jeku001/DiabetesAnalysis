import pickle
import numpy as np

class DiabetesPredictionApp:
    def __init__(self, model_path):
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, input_data):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = self.model.predict(input_array)
        return "Diabetes" if prediction[0] == 1 else "No Diabetes"

if __name__ == "__main__":
    print("Provide the following information:")
    HighBP = int(input("HighBP (0/1): "))
    HighChol = int(input("HighChol (0/1): "))
    CholCheck = int(input("CholCheck (0/1): "))
    BMI = float(input("BMI: "))
    Smoker = int(input("Smoker (0/1): "))
    Stroke = int(input("Stroke (0/1): "))
    HeartDiseaseorAttack = int(input("HeartDiseaseorAttack (0/1): "))
    PhysActivity = int(input("PhysActivity (0/1): "))
    HvyAlcoholConsump = int(input("HvyAlcoholConsump (0/1): "))
    GenHlth = float(input("GenHlth (0-5): "))
    MentHlth = float(input("MentHlth (number of days): "))
    DiffWalk = int(input("DiffWalk (0/1): "))
    Sex = int(input("Sex (0 for female, 1 for male): "))
    Age = int(input("Age (integer): "))
    Income = float(input("Income (1-8 scale): "))

    app = DiabetesPredictionApp("logistic_model.pkl")
    input_data = [
        HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
        HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, GenHlth,
        MentHlth, DiffWalk, Sex, Age, Income
    ]

    result = app.predict(input_data)
    print(f"Prediction Result: {result}")
