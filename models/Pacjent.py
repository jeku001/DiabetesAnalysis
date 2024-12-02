class Pacjent:
    def __init__(self, wiek, bmi, wzrost, high_bp):
        self.wiek = wiek
        self.bmi = bmi
        self.wzrost = wzrost
        self.high_bp = high_bp

    def show_info(self):
        return (
            f"Wiek: {self.wiek}, BMI: {self.bmi}, Wzrost: {self.wzrost} cm, "
            f"Nadciśnienie: {'Tak' if self.high_bp else 'Nie'}"
        )


