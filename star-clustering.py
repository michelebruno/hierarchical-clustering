import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
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

    distance = np.cos(np.deg2rad(90 - dec1)) * np.cos(np.deg2rad(90 - dec2)) + np.sin(np.deg2rad(90 - dec1)) * np.sin(np.deg2rad(90 - dec2)) * np.cos(np.deg2rad(ra1 - ra2))

    if distance > 1:
        distance = 1

    distance = np.degrees(np.arccos(distance))
    if distance < 0:
        print("Negative distance")

    return distance


df = pd.read_csv(r'database_root.csv', low_memory=False)

# Milano è 45°28′01″N 9°11′24″E

print(df.shape)

df = df[df["dec"] > 0]

unnamed = df[pd.isnull(df['proper'])]
unnamed = unnamed[unnamed["mag"] < 7]

# df = df[df["mag"] < 4.5]
df = df[~pd.isnull(df['proper'])]
# df["proper"].fillna('no name')
print(df.shape)

distance_matrix = sch.linkage(df.loc[:, ["ra", "dec"]], 'single', sphere_distance)
figure = plt.figure(figsize=(25, 10))
dn = sch.dendrogram(distance_matrix)
figure.show()

ac = AgglomerativeClustering(
    n_clusters=24,
    affinity=lambda X: pairwise_distances(X, metric=sphere_distance),
    linkage='single'
)

ac.fit(df.loc[:, ["ra", "dec"]])

# Associa a ciascun index dei dati, il cluster di appartenenza
clustered_data = pd.DataFrame([df.index, ac.labels_]).T

# fig.add_trace(go.Scatterpolar(
#     r=df['dec'],
#     theta=[datum['ra'] * 360 / 24 for index, datum in df.iterrows()],
#     mode='markers',
#     name='Figure 8',
#     marker=dict(
#         color="red",
#         # symbol="square",
#         size=8
#     )
# ))

colors = [
    "red",
    "green",
    "yellow"
]

grouped_indexes = clustered_data.groupby(1)

# Inizializza la figura
fig = go.Figure()

# fig.add_trace(go.Scatterpolar(
#     r=unnamed['dec'],
#     theta=[datum['ra'] * 360 / 24 for index, datum in unnamed.iterrows()],
#     mode='markers',
#     text=unnamed["proper"],
#     marker=dict(
#         # color=colors[label],
#         # symbol="square",
#         size=3, opacity=.5,
#     )
# ))

for label in range(grouped_indexes.ngroups):
    indexes = grouped_indexes.groups[label]

    filtered = df.iloc[indexes]

    # print(filtered['dec'])
    fig.add_trace(go.Scatterpolar(
        r=filtered['dec'],
        theta=[datum['ra'] * 360 / 24 for index, datum in filtered.iterrows()],
        mode='markers',
        text=filtered["proper"],
        marker=dict(
            # color=colors[label],
            # symbol="square",
            size=5
        )
    ))

#


fig.update_layout(polar=dict(
    radialaxis=dict(range=[90, 0]),
    # angularaxis=dict(showticklabels=False, ticks='')
))

fig.show()

if not os.path.exists("images"):
    os.mkdir("images")

fig.write_image("images/fig1.svg", engine="kaleido")

#
# data = df.values
#
# plt.figure(figsize=(10, 7))
# plt.title("Customer Dendrograms")
#
# dend = sch.dendrogram(sch.linkage(data, method='single'))
#
# plt.show()
