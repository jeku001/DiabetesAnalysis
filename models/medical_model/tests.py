import pandas as pd
from Describe_data import DescribeData
from Train import TrainAndEvaluate


"""
data = pd.read_csv("../../data/medical_data/diabetes.csv")
outcome_counts = data['Outcome'].value_counts()
print(outcome_counts)

data = pd.read_pickle("processed_data.pkl")
outcome_counts = data['Outcome'].value_counts()
print(outcome_counts)

"""

"""
data = pd.read_pickle("processed_data.pkl")
for i in range(1, 7):
    trainer = TrainAndEvaluate(data, 'Outcome')
    trainer.train()
    
  """


data = pd.read_pickle("processed_data.pkl")
print(data.columns)









