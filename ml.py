import pandas as pd

path = "./tic-tac-toe.txt"
df = pd.read_csv(path,
                 header=None,
                 encoding="utf-8"
)
print(df.head())
