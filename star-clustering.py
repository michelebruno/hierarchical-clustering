import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# pd.options.plotting.backend = "plotly"

df = pd.read_csv(r'database_NorthernEmisphere.csv', low_memory=False)

# Selected columns
df_sl = df.iloc[:, 7:9]
# Inizializza la figura
fig = go.Figure()

ac = AgglomerativeClustering(
    n_clusters=12
)

ac.fit(df_sl.values)

# Associa a ciascun index dei dati, il cluster di appartenenza
clustered_data = pd.DataFrame([df_sl.index, ac.labels_]).T

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

]

grouped_indexes = clustered_data.groupby(1)

for label in range(grouped_indexes.ngroups):
    indexes = grouped_indexes.groups[label]

    filtered = df.iloc[indexes]

    # print(filtered['dec'])
    fig.add_trace(go.Scatterpolar(
        r=filtered['dec'],
        theta=[datum['ra'] * 360 / 24 for index, datum in filtered.iterrows()],
        mode='markers',
        name='Figure 8',
        marker=dict(
            color=np.random.choice(range(256), 3),
            # symbol="square",
            size=3
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

# fig.write_image("images/fig1.svg", engine="kaleido")

#
# data = df.values
#
# plt.figure(figsize=(10, 7))
# plt.title("Customer Dendrograms")
#
# dend = sch.dendrogram(sch.linkage(data, method='single'))
#
# plt.show()
