# Stub
class PacjentStub:
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

# test STUB
def test_pacjent_show_info_stub():
    pacjent = PacjentStub(wiek=40, bmi=25, wzrost=170, high_bp=False)
    expected_result = "Wiek: 40, BMI: 25, Wzrost: 170 cm, Nadciśnienie: Nie"
    assert pacjent.show_info() == expected_result, "Niepoprawny wynik show_info()"

# Uruchomienie testu
test_pacjent_show_info_stub()
