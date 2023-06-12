import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv('data/data.csv')
integer_cols = df.select_dtypes(include=['int64']).columns

scaler = StandardScaler()
df[integer_cols] = scaler.fit_transform(df[integer_cols])

df.to_csv('data/data_normalized.csv', index=False)

plot = sns.violinplot(data=df[integer_cols])
fig = plot.get_figure()
fig.savefig('plots/normalized_integers.png')