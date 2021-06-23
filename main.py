import pandas as pd

df = pd.read_csv("titanic/data/gender_submission.csv")

print(df.to_string(index=False))