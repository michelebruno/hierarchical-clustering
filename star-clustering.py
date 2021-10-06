import os
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
from plotly.figure_factory import create_dendrogram


# calculates the distance on a sphere, ra and dec must be in degrees
def sphere_distance(p1, p2):
    """
    Calculates the distance on a sphere, ra and dec must be in degrees
    """
    ra1 = p1[0] * 360 / 24
    dec1 = p1[1]
    ra2 = p2[0] * 360 / 24
    dec2 = p2[1]

    # Turn vars into radians
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)

    # 90 degrees in radians
    rect = np.deg2rad(90)

    distance = np.cos(rect - dec1) * np.cos(rect - dec2) + np.sin(rect - dec1) * np.sin(
        rect - dec2) * np.cos(ra1 - ra2)

    # In some cases, when calculating the distance of a point from itself,
    # distance equals to 1.0000000000000002 for a rounding error
    if distance > 1:
        # print("This distance is gte than 1.",distance, p1,p2)
        distance = 1

    distance = np.degrees(np.arccos(distance))

    if distance < 0:
        print("Negative distance")

    return distance


def calc_dist_by_id(id1, id2):
    """
    Prints geodesic distance. For console usage.
    """
    point1 = df.loc[df['id'] == id1, ["ra", "dec"]].iloc[0]
    point2 = df.loc[df['id'] == id2, ["ra", "dec"]].iloc[0]

    print(sphere_distance(point1, point2))


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


df = pd.read_csv(r'database_root.csv', low_memory=False)

# Milano è 45°28′01″N 9°11′24″E

print("Initial shape: ", df.shape)

# Get only stars above equator
df = df[df["dec"] > 0]

# Assign to unnmaed only stars with no name and mag lower than a given value
unnamed = df[pd.isnull(df['proper'])]
unnamed = unnamed[unnamed["mag"] < 7]

# Keep only named stars
df = df[~pd.isnull(df['proper'])]

print("Filtered named star: ", df.shape)

# Concat named stars with brightest unnamed stars
df = pd.concat([df, unnamed[unnamed["mag"] < 3.5]])

print("Final shape: ", df.shape)

sizes = []

# Loop each star and get the size in plot based on magnitude
for index, item in df.iterrows():
    size = scale(
        item["mag"],
        [max(df["mag"]), min(df["mag"])],
        [.05, 4]
    )
    sizes.append(size)

# Add size column to each record
df.insert(0, "size", sizes)

# Max star distance to define sub-clusters
distance_threshold = 12

# Create an instance of AgglomerativeClustering
ac = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=distance_threshold,
    affinity=lambda X: pairwise_distances(X, metric=sphere_distance),
    linkage='single'
)

# TODO Differenza tra fit e fit_predict?
ac.fit(df.loc[:, ["ra", "dec"]])

print(max(ac.labels_ + 1), "constellations found.")

# Associa a ciascun index dei dati, il cluster di appartenenza
clustered_data = pd.DataFrame([df.index, ac.labels_]).T

# Groups stars based on assigned cluster
grouped_indexes = clustered_data.groupby(1)

# Initialized scatter plot
fig = go.Figure()
for label in range(grouped_indexes.ngroups):
    indexes = grouped_indexes.groups[label]

    filtered = df.iloc[indexes]

    fig.add_trace(go.Scatterpolar(
        r=filtered['dec'],
        theta=[datum['ra'] * 360 / 24 for index, datum in filtered.iterrows()],
        mode='markers',
        text=[str(item["id"]) + " " + str(item["proper"]) for i, item in filtered.iterrows()],

        marker=dict(
            color = "white",
            opacity=1,
            #size=filtered["size"],
            size=2,
            line=dict(
                width=0
            )
        )
    ))

fig.update_layout(polar=dict(
    # Inverte l'asse dec
    radialaxis=dict(range=[90, 0]),
    bgcolor='#384554',
))

# Shows scatter polar
fig.show()

if not os.path.exists("images"):
    os.mkdir("images")

# Saves svg
fig.write_image("images/scatterpolar.svg")

# Dendrogram
dendro = create_dendrogram(
    df.loc[:, ["ra", "dec"]],
    color_threshold=distance_threshold,
    distfun=lambda x: pdist(x, metric=sphere_distance),
    linkagefun=lambda x: sch.linkage(x, "single"),
    labels=[item["proper"] if not pd.isnull(item["proper"]) else item["id"] for i, item in df.iterrows()]
)

dendro.update_layout({'width': 1400, 'height': 900})
dendro.show()
dendro.write_image("images/dendrogram.svg")
