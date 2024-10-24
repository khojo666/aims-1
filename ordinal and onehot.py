import pandas as pd
import numpy as np

data = {
    'Animal': ['Cat', 'Dog', 'Bird', 'Cat', 'Dog'],
    'Color': ['Red', 'Blue', 'Green', 'Green', 'Red'],
}

df = pd.DataFrame(data)

def ordinal_encode(df, column):
    unique_vals = df[column].unique()
    ordinal_mapping = {val: idx for idx, val in enumerate(unique_vals)}
    df[column] = df[column].map(ordinal_mapping)
    return ordinal_mapping

animal_mapping = ordinal_encode(df, 'Animal')
print("Ordinal Encoding for 'Animal':")
print(df)
print("Mapping:", animal_mapping)

def one_hot_encode(df, column):
    unique_vals = df[column].unique()
    for val in unique_vals:
        df[f'{column}_{val}'] = np.where(df[column] == val, 1, 0)
    df.drop(column, axis=1, inplace=True)

one_hot_encode(df, 'Color')
print("\nOne-Hot Encoding for 'Color':")
print(df)