import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DescribeData:
    def __init__(self, data, output_dir='output'):
        self.output_dir = output_dir
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            if data.endswith('.csv'):
                self.data = pd.read_csv(data)
            elif data.endswith('.pkl'):
                self.data = pd.read_pickle(data)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or PKL file.")
        else:
            raise ValueError("Data input should be a DataFrame, a CSV file path, or a PKL file path.")

    def basic_statistics(self):
        return self.data.describe()

    def advanced_statistics(self):
        stats = pd.DataFrame()
        stats['mean'] = self.data.mean()
        stats['median'] = self.data.median()
        stats['skew'] = self.data.skew()
        stats['kurtosis'] = self.data.kurt()
        stats['variance'] = self.data.var()
        stats['std_dev'] = self.data.std()
        return stats

    def correlation_matrix(self):
        return self.data.corr()

    def plot_correlation_matrix(self):
        """ Generate and save a heatmap of the correlation matrix. """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.correlation_matrix(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(f'{self.output_dir}/correlation_matrix.png')
        plt.close()

    def describe(self):
        """ Generate all descriptions and save correlation matrix plot. """
        descriptions = {
            'basic_stats': self.basic_statistics(),
            'advanced_stats': self.advanced_statistics(),
            'correlation_matrix': self.correlation_matrix(),
        }
        self.plot_correlation_matrix()  # Automatically save the plot
        return descriptions

