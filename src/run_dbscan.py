from sklearn.cluster import DBSCAN
import pandas as pd
import yaml
import numpy as np

# Generate sample data
df = pd.read_csv('data/data_normalized.csv')

params = yaml.safe_load(open('params.yaml'))
eps = params['dbscan']['eps']
min_samples = params['dbscan']['min_samples']

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(df)

# Save labels to pickle file
np.save('results/dbscan.npy', labels)