import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances


# pd.options.plotting.backend = "plotly"

# calculates the distance on a sphere, ra and dec must be in degrees
def sphere_distance(p1, p2):
    ra1 = p1[0] * 360 / 24
    dec1 = p1[1]
    ra2 = p2[0] * 360 / 24
    dec2 = p2[1]

    distance = np.cos(np.deg2rad(90 - dec1)) * np.cos(np.deg2rad(90 - dec2)) + np.sin(np.deg2rad(90 - dec1)) * np.sin(
        np.deg2rad(90 - dec2)) * np.cos(np.deg2rad(ra1 - ra2))

    if distance > 1:
        distance = 1

    distance = np.degrees(np.arccos(distance))
    if distance < 0:
        print("Negative distance")

    return distance


def calc_dist_by_id(id1, id2):
    point1 = df.loc[df['id'] == id1, ["ra", "dec"]].iloc[0]
    point2 = df.loc[df['id'] == id2, ["ra", "dec"]].iloc[0]

    print(sphere_distance(point1, point2))


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


df = pd.read_csv(r'database_root.csv', low_memory=False)

# Milano è 45°28′01″N 9°11′24″E

print("Initial shape: ", df.shape)

df = df[df["dec"] > 0]

unnamed = df[pd.isnull(df['proper'])]
unnamed = unnamed[unnamed["mag"] < 7]

df = df[~pd.isnull(df['proper'])]
# df["proper"].fillna('no name')
# df = df[df["mag"] < 6]


print("Filtered named star: ", df.shape)

df = pd.concat([df, unnamed[unnamed["mag"] < 3.5]])

print("Final shape: ", df.shape)

# print(filtered["mag"])
sizes = []
opacities = []

for index, item in df.iterrows():
    if item["mag"] < 3:
        sizes.append(10)
        opacities.append(1)
    else:
        sizes.append(8)
        opacities.append(.8)

df.insert(0, "opacity", opacities)
df.insert(0, "size", sizes)

ac = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=12,
    compute_full_tree=True,
    compute_distances=True,
    affinity=lambda X: pairwise_distances(X, metric=sphere_distance),
    linkage='single'
)

# TODO Differenza tra fit e fit_predict?
ac.fit(df.loc[:, ["ra", "dec"]])

print(max(ac.labels_ + 1), "costellazioni")

figure = plt.figure(figsize=(80, 40))
plot_dendrogram(ac, leaf_label_func= lambda x: df.iloc[x]["proper"] if df.iloc[x]["proper"] else df.iloc[x]["id"])
# plt.savefig("images/dendrogram.svg")
plt.show()

# Associa a ciascun index dei dati, il cluster di appartenenza
clustered_data = pd.DataFrame([df.index, ac.labels_]).T

colors = [
    "red",
    "green",
    "yellow"
]

grouped_indexes = clustered_data.groupby(1)

# Inizializza la figura
fig = go.Figure()

for label in range(grouped_indexes.ngroups):
    indexes = grouped_indexes.groups[label]

    filtered = df.iloc[indexes]

    # filtered["mag"].apply(lambda x: float(x))

    fig.add_trace(go.Scatterpolar(
        r=filtered['dec'],
        theta=[datum['ra'] * 360 / 24 for index, datum in filtered.iterrows()],
        mode='markers',
        text=filtered["id"],
        marker=dict(
            # color=colors[label],
            # symbol="square",
            opacity=filtered["opacity"],
            # size=5 - np.log(4 - filtered["mag"]),
            size=filtered["size"]
        )
    ))

fig.update_layout(polar=dict(
    # Inverte l'asse dec
    radialaxis=dict(range=[90, 0]),
    # angularaxis=dict(showticklabels=False, ticks='')
))

fig.show()

if not os.path.exists("images"):
    os.mkdir("images")

fig.write_image("images/fig1.svg")
