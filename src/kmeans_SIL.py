from sklearn.cluster import KMeans
import json
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

params = yaml.safe_load(open('params.yaml'))

df = pd.read_csv('data/data_normalized.csv')

with open('metrics/sil_scores.json') as f:
    json_data=json.load(f)
scores=[json_data['sil_scores'][x]['sil_score'] for x in range(len(json_data['sil_scores']))] 
scores=[(x-min(scores))/(max(scores)-min(scores)) for x in scores]

with open('metrics/db_scores.json') as f:
    json_data=json.load(f)
    
scoresdb=[json_data['db_scores'][x]['db_score'] for x in range(len(json_data['db_scores']))] 
scoresdb=[(x-min(scoresdb))/(max(scoresdb)-min(scoresdb)) for x in scoresdb]
combined_score=[scores[x]-scoresdb[x]+(x/250) for x in range(len(scores))]

sns.lineplot(y=combined_score, x=range(2,50))
plt.savefig('plots/combined_score.png')

best_k=[index+2 for index, value in sorted(enumerate(combined_score), key=lambda x: x[1])[:3]] + [5,7,10]

kmeans_metrics = []
for k in best_k:
    print("k: ", k)
    model=KMeans(n_clusters=k, n_init="auto", random_state=params['seed'])
    labels=model.fit_predict(df)
    
    print("Calculating cosine similarity")
    cos_sim=pairwise.cosine_similarity(model.cluster_centers_)
    avg_cos_sim=np.mean(cos_sim)
    sns.heatmap(cos_sim,vmin=-1,vmax=1)
    plt.savefig('plots/cos_sim_K:'+str(k))
    plt.clf()

    print("Calculating euclidean distance")
    eucl_dist=pairwise.euclidean_distances(model.cluster_centers_)
    sns.heatmap(eucl_dist,vmin=0)
    plt.savefig('plots/eucl_dist_K:'+str(k))
    plt.clf()
    avg_eucl_dist=np.mean(eucl_dist)

    print("Calculating manhattan distance")
    man_dist=pairwise.manhattan_distances(model.cluster_centers_)
    sns.heatmap(man_dist,vmin=0)
    plt.savefig('plots/man_dist_K:'+str(k))
    plt.clf()
    avg_man_dist=np.mean(man_dist)
    
    print("Calculating davies bouldin")
    db_score = davies_bouldin_score(df, labels)
    
    kmeans_metrics.append([
        k,
        db_score,
        avg_cos_sim,
        avg_eucl_dist])

    np.savetxt(f"metrics/cosine_sims_{k}.txt", cos_sim, fmt="%s")
    np.savetxt(f"metrics/eucl_dists_{k}.txt", eucl_dist, fmt="%s")
    np.savetxt(f"metrics/man_dists_{k}.txt", man_dist, fmt="%s")

metrics_df = pd.DataFrame(kmeans_metrics, columns=['k', 'db_score', 'avg_cos_sim', 'avg_eucl_dist'])
metrics_df.to_csv('metrics/kmeans_metrics.csv', index=False)

