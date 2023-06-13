from sklearn.cluster import KMeans
import json
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
import numpy as np
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

cosine_sims={'cosine_sims': []}
eucl_dists={'eucl_dists': []}
man_dists={'man_dists': []}

best_k=[index+2 for index, value in sorted(enumerate(combined_score), key=lambda x: x[1])[:3]] + [5,10]

for k in best_k:
    model=KMeans(n_clusters=k, n_init="auto", random_state=params['seed'])
    labels=model.fit_predict(df)
    cos_sim=pairwise.cosine_similarity(model.cluster_centers_)
    cosine_sims['cosine_sims'].append({'k': k, 'cos_sim': cos_sim})
    sns.heatmap(cos_sim,vmin=-1,vmax=1)
    plt.savefig('plots/cos_sim_K:'+str(k))
    plt.clf()
    eucl_dist=pairwise.euclidean_distances(model.cluster_centers_)
    sns.heatmap(eucl_dist,vmin=0)
    plt.savefig('plots/eucl_dist_K:'+str(k))
    plt.clf()
    eucl_dists['eucl_dists'].append({'k': k, 'eucl_dist': eucl_dist})
    man_dist=pairwise.manhattan_distances(model.cluster_centers_)
    sns.heatmap(man_dist,vmin=0)
    plt.savefig('plots/man_dist_K:'+str(k))
    plt.clf()
    man_dists['man_dists'].append({'k': k, 'man_dist': man_dist})

with open("metrics/cosine_sims.json", "w") as f:
    json.dump(cosine_sims, f)
with open("metrics/eucl_dists.json", "w") as f:
    json.dump(eucl_dists, f)
with open("metrics/man_dists.json", "w") as f:
    json.dump(man_dists, f)