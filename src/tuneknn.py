import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import yaml
import json

params = yaml.safe_load(open('params.yaml'))

df = pd.read_csv('data/data_normalized.csv')

start_k = params['tuneknn']['min_k']
end_k = params['tuneknn']['max_k']
sil_scores = {"sil_scores": []} 
db_scores = {"db_scores": []}

seed = params['seed']
for k in range(start_k, end_k, 1):
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = kmeans.fit_predict(df)
    sil_score = silhouette_score(df, labels, metric='euclidean')
    db_score = davies_bouldin_score(df, labels)
    sil_scores['sil_scores'].append({
        'k': k,
        'sil_score': sil_score
    })

    db_scores['db_scores'].append({
        'k': k,
        'db_score': db_score
    })


with open("metrics/sil_scores.json", "w") as f:
    json.dump(sil_scores, f)

with open("metrics/db_scores.json", "w") as f:
    json.dump(db_scores, f)