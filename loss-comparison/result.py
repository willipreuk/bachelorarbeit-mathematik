import pandas as pd

csv_file_path = 'results-b.csv'

df = pd.read_csv(csv_file_path).groupby("Loss Function")
df = df.agg({
    'Test Loss': ['mean', 'median', 'std'],
    'Test MAE': ['mean', 'median', 'std'],
    'I_trapez - I_pred': ['mean', 'median', 'std'],
    'I_ref - I_pred': ['mean', 'median', 'std'],
    'I_ref - I_trapez': ['mean']
})
df.sort_values(('I_ref - I_pred', 'mean'), inplace=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df)
