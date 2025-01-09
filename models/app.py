from Predict import Predict
import numpy as np

def get_input_data(fields):
    return [input(f"{field}: ") for field in fields]


if __name__ == "__main__":
    print("Please select the model type: 'survey' or 'medical'")
    model_type = input("Enter model type: ").strip().lower()

    if model_type == 'survey':
        print("Provide the following information:")
        fields = [
            "HighBP (0/1)",
            "HighChol (0/1)",
            "CholCheck (0/1)",
            "BMI",
            "Smoker (0/1)",
            "Stroke (0/1)",
            "HeartDiseaseorAttack (0/1)",
            "PhysActivity (0/1)",
            "HvyAlcoholConsump (0/1)",
            "GenHlth (0-5)",
            "MentHlth (number of days)",
            "DiffWalk (0/1)",
            "Sex (0 for female, 1 for male)",
            "Age (integer)",
            "Income (1-8 scale)"
        ]
        input_data = get_input_data(fields)

    elif model_type == 'medical':
        print("Provide the following medical information:")
        fields = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
        input_data = get_input_data(fields)

    else:
        raise ValueError("Invalid model type selected. Please choose 'survey' or 'medical'.")

    input_data = [float(x) if '.' in x or x in ['BMI', 'DiabetesPedigreeFunction'] else int(x) if x else np.nan for x in
                  input_data]
    print(input_data) #debug
    predictor = Predict(model_type=model_type, input_data=input_data)
    result = predictor.predict()
    print(f"Prediction Result: {result}")
