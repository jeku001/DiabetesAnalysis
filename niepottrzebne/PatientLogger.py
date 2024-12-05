class PatientLogger:
    def log_patient_info(self, patient):
        info = patient.show_info()
        print(f"Logging patient info: {info}")
        return info