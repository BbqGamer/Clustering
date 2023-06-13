from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import pandas as pd
import sys
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
# Generate sample data
df = pd.read_csv('./data/data_normalized.csv')

with open('metrics/best_eps.json') as f:
    json_data=json.load(f)
params=json_data['best_eps']
eps=[x['eps'] for x in params]
min_samples=[x['k'] for x in params]
# Get parameters from command line eps and min_samples

for x in range(len(eps)):
    dbscan = DBSCAN(eps=eps[x], min_samples=min_samples[x])
    labels = dbscan.fit_predict(df)
    num_clusters = np.unique(labels).shape[0]
    

    db_score = davies_bouldin_score(df, labels) 
    
    print('start')
    clusters = {}
    for label in labels:
        if label != -1:  # Exclude noise points
            cluster_points = df[dbscan.labels_ == label]
            clusters[label] = cluster_points
    print('clustes')
# Compute centroid/representative vector for each cluster
    centroids = []
    for cluster_points in clusters.values():
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    print('finished centroids')
    cos_sim=pairwise.cosine_similarity(centroids)
    sns.heatmap(cos_sim,vmin=-1,vmax=1)
    plt.savefig('plots/cos_sim_samples:'+str(min_samples[x]))
    plt.clf()
    eucl_dist=pairwise.euclidean_distances(centroids)
    sns.heatmap(eucl_dist,vmin=0)
    plt.savefig('plots/eucl_dist_samples:'+str(min_samples[x]))
    plt.clf()

    man_dist=pairwise.manhattan_distances(centroids)
    sns.heatmap(man_dist,vmin=0)
    plt.savefig('plots/man_dist_samples:'+str(min_samples[x]))
    plt.clf()

    print(f"eps: {eps[x]}, min_samples: {min_samples[x]}, number of clusters: {num_clusters} db_score: {db_score}")