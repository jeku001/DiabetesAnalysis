import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesDataAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_pickle(file_path)

    def summary_statistics(self):
        print("Summary Statistics:")
        print(self.data.describe())

    def missing_values(self):
        print("Missing Values:")
        print(self.data.isnull().sum())

    def visualize_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x="Diabetes_binary", data=self.data)
        plt.title("Distribution of Diabetes")
        plt.show()

    def correlation_heatmap(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

if __name__ == "__main__":
    analysis = DiabetesDataAnalysis("df_binary_reduced.pkl")
    analysis.summary_statistics()
    analysis.missing_values()
    analysis.visualize_distribution()
    analysis.correlation_heatmap()
