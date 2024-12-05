from unittest.mock import Mock
import Patient
import PatientLogger
def test_patient_show_info_mock():
    mock_patient = Mock(spec=Patient)
    mock_patient.show_info.return_value = "Wiek: 40, BMI: 25, Wzrost: 170 cm, Nadciśnienie: Tak"
    logger = PatientLogger()
    result = logger.log_patient_info(mock_patient)
    mock_patient.show_info.assert_called_once()

    assert result == "Wiek: 40, BMI: 25, Wzrost: 170 cm, Nadciśnienie: Tak", "Zwrócono niepoprawny wynik"
