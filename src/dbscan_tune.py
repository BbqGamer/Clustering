from sklearn.neighbors import NearestNeighbors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('./data/data_normalized.csv')
nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(df)
distances, indices = neighbors.kneighbors(df)
distances = np.sort(distances[:,10], axis=0)
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
plt.savefig('plots/dbscan_knee_10samples.png')
print(distances[knee.knee])


nearest_neighbors = NearestNeighbors(n_neighbors=6)
neighbors = nearest_neighbors.fit(df)
distances, indices = neighbors.kneighbors(df)
distances = np.sort(distances[:,5], axis=0)
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
plt.savefig('plots/dbscan_knee_5samples.png')
print(distances[knee.knee])