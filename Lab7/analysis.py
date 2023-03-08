import pandas as pd

# Read in the data
df = pd.read_csv('data.csv')


print(df.groupby(['alpha'])['best_cost'].mean())
print(df.groupby(['beta'])['best_cost'].mean())
print(df.groupby(['reduction_rate'])['best_cost'].mean())
print(df.groupby(['k_update'])['best_cost'].mean())
print(df.groupby(['colony_size'])['best_cost'].mean())
