
from survey_model.Predict import Predict


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

    input_data = [
        HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
        HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, GenHlth,
        MentHlth, DiffWalk, Sex, Age, Income
    ]

    #
    predictor = Predict(model_type='survey', input_data=input_data)
    result = predictor.predict()
    print(f"Prediction Result: {result}")
