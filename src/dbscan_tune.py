from sklearn.neighbors import NearestNeighbors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
best_eps={'best_eps':[]}
for x in [3,4,5,8,10]:
    df = pd.read_csv('./data/data_normalized.csv')
    nearest_neighbors = NearestNeighbors(n_neighbors=x+1)
    neighbors = nearest_neighbors.fit(df)
    distances, indices = neighbors.kneighbors(df)
    distances = np.sort(distances[:,x], axis=0)
    fig = plt.figure(figsize=(5, 5))

    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance")
    from kneed import KneeLocator
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    fig = plt.figure(figsize=(5, 5))
    knee.plot_knee()
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.savefig('plots/dbscan_knee_'+str(x)+'samples.png')
    best_eps['best_eps'].append({'k': x, 'eps': distances[knee.knee]})


with open("metrics/best_eps.json", "w") as f:
    json.dump(best_eps, f)