import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import json
import time

# Get parameters from command line eps and min_samples
if len(sys.argv) != 2:
    print("Usage: python evaluate.py [labels_file]")
    sys.exit(1)
    
labels_file = sys.argv[1]

# Generate sample data
df = pd.read_csv('data/data_normalized.csv')
labels = np.load(labels_file)

# Calculate basic scores
num_clusters = np.unique(labels).shape[0]

if num_clusters > 1:
    print("Calculating silhouette score...")
    sil_score = silhouette_score(df, labels)
    sil_score = -1
else:
    print("Not enough clusters to calculate silhouette score")
    sil_score = -1

print("Calculating Davies-Bouldin score...")
db_score = davies_bouldin_score(df, labels)

results = {
    'num_clusters': num_clusters,
    'sil_score': sil_score,
    'db_score': db_score
}

with open('metrics/evaluation.json', 'w') as f:
    json.dump(results, f)

print("Running PCA - reducing to 3D...")
pca = PCA(n_components=3)
pca.fit(df)
pca_df = pca.transform(df)
reduced_df = pd.DataFrame(pca_df, columns=['pca1', 'pca2', 'pca3'])

print("Plotting 3D clusters...")
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(reduced_df['pca1'], reduced_df['pca2'], reduced_df['pca3'], c = labels)# save to file
plt.savefig('plots/3d_clusters.png')

print("Running PCA - reducing to 2D...")
pca = PCA(n_components=2)
pca.fit(df)
pca_df = pca.transform(df)

print("Plotting 2D clusters...")
reduced_df = pd.DataFrame(pca_df, columns=['pca1', 'pca2'])
plt.figure(figsize = (12,7))
sns.scatterplot(data = reduced_df, x = 'pca1', y = 'pca2',  hue = labels, palette = 'Set2')
plt.savefig('plots/2d_clusters.png')