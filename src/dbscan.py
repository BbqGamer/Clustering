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
metrics=[]
for x in range(len(eps)):
    for p in range(2):
        if p==1:
            
            dbscan = DBSCAN(eps=4, min_samples=min_samples[x])
        else:
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
        avg_cos_sim=np.mean(cos_sim)
        sns.heatmap(cos_sim,vmin=-1,vmax=1)
        plt.savefig('plots/cos_sim_samples:'+str(min_samples[x])+'eps:'+str(p))
        plt.clf()
        eucl_dist=pairwise.euclidean_distances(centroids)
        avg_eucl_dist=np.mean(eucl_dist)
        sns.heatmap(eucl_dist,vmin=0)
        plt.savefig('plots/eucl_dist_samples:'+str(min_samples[x])+'eps:'+str(p))
        plt.clf()

        man_dist=pairwise.manhattan_distances(centroids)
        sns.heatmap(man_dist,vmin=0)
        plt.savefig('plots/man_dist_samples:'+str(min_samples[x])+'eps:'+str(p))
        plt.clf()
        avg_man_dist=np.mean(man_dist)
        if p==1:
            metrics.append([
                min_samples[x],
                4,
                db_score,
                avg_cos_sim,
                avg_eucl_dist,
                avg_man_dist])
        else:
            metrics.append([
                min_samples[x],
                eps[x],
                db_score,
                avg_cos_sim,
                avg_eucl_dist,
                avg_man_dist])
        
        np.savetxt(f"metrics/cosine_sims_sample{min_samples[x]}.txt", cos_sim, fmt="%s")
        np.savetxt(f"metrics/eucl_dists_sample{min_samples[x]}.txt", eucl_dist, fmt="%s")
        np.savetxt(f"metrics/man_dists_sample{min_samples[x]}.txt", man_dist, fmt="%s")
        

metrics_df = pd.DataFrame(metrics, columns=['min_samples','eps', 'db_score', 'avg_cos_sim', 'avg_eucl_dist', 'avg_man_dist'])
metrics_df.to_csv('metrics/kmeans_metrics.csv', index=False)