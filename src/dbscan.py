from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import pandas as pd
import sys

# Generate sample data
df = pd.read_csv('../data/data_normalized.csv')

# Get parameters from command line eps and min_samples
if len(sys.argv) != 3:
    print("Usage: python dbscan.py [eps] [min_samples]")
    sys.exit(1)
    
eps = float(sys.argv[1])
min_samples = int(sys.argv[2])

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(df)

num_clusters = np.unique(labels).shape[0]
if num_clusters > 1:
    sil_score = silhouette_score(df, labels)
else:
    sil_score = -1

db_score = davies_bouldin_score(df, labels) 


print(f"eps: {eps}, min_samples: {min_samples}, number of clusters: {num_clusters} silhouette score: {sil_score}, db_score: {db_score}")