import pandas as pd

df = pd.read_csv("dataset/health_data.csv")

print("Columns:")
print(list(df.columns))

print("\nNumber of columns:", len(df.columns))
print("\nFirst 5 rows:")
print(df.head())