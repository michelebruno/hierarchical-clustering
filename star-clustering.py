import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import kaleido

# pd.options.plotting.backend = "plotly"

df = pd.read_csv(r'database_NorthernEmisphere.csv', low_memory=False, usecols=["ra", "dec"])

print(df.shape)
fig = go.Figure(

)

fig.add_trace(go.Scatterpolar(
    r=df['dec'],
    theta=[datum['ra'] * 360 / 24 for index, datum in df.iterrows()],
    mode='markers',
    name='Figure 8',

    marker=dict(
        color="red",
        # symbol="square",
        size=8
    )
))

fig.update_layout(polar=dict(
    radialaxis=dict(range=[90, 0]),
    # angularaxis=dict(showticklabels=False, ticks='')
))

# fig.show()

if not os.path.exists("images"):
    os.mkdir("images")

fig.write_image("images/fig1.svg",  engine="kaleido")

#
# data = df.values
#
# plt.figure(figsize=(10, 7))
# plt.title("Customer Dendrograms")
#
# dend = sch.dendrogram(sch.linkage(data, method='single'))
#
# plt.show()
