import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np


def load_data(filename):
    return pd.read_pickle(filename)

def balance_data_with_smote(df):
    smote = SMOTE(random_state=42)
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    X_sm, y_sm = smote.fit_resample(X, y)
    return pd.concat([pd.DataFrame(y_sm), pd.DataFrame(X_sm, columns=X.columns)], axis=1)

def add_missing_data(df, fraction=0.5):
    """
    Powiela określoną część (fraction) wierszy, dodając braki danych w losowych kolumnach w losowej liczbie od 2 do 8
    """
    rows_to_duplicate = np.random.choice(df.index, size=int(len(df) * fraction), replace=False)
    duplicated_rows = df.loc[rows_to_duplicate].copy()

    columns_to_consider = df.columns.drop('Diabetes_binary')  # Wykluczenie kolumny 'Diabetes_binary'

    for row in duplicated_rows.index:
        num_cols = np.random.choice(range(2, 9))
        cols_to_nan = np.random.choice(columns_to_consider, size=num_cols, replace=False)
        duplicated_rows.loc[row, cols_to_nan] = np.nan

    combined_df = pd.concat([df, duplicated_rows])

    final_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return final_df


def main():
    df = load_data('df_binary_reduced.pkl')
    balanced_df = balance_data_with_smote(df)
    final_df = add_missing_data(balanced_df)
    final_df.to_pickle('processed_data.pkl')


if __name__ == "__main__":
    main()