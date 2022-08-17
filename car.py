# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
import seaborn as sns
import numpy as np

# %% #label
df = pd.read_csv('/Users/caohaha/Downloads/car.txt', header=None)
df.dropna(inplace=True)
cat_columns = df.select_dtypes(['object']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
categorical = df[6]
projection = TSNE().fit_transform(df)
color_palette = sns.color_palette("husl", 20)
cluster_colors = [color_palette[x - 3] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in categorical.values]
cluster_member_colors = cluster_colors
plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.title("Category")
plt.show()
# %% HDBSCAN
df['label'] = categorical
clusterer = hdbscan.HDBSCAN(min_cluster_size=3).fit(df)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
clusters = len(np.unique(clusterer.labels_))
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.title("HDBSCAN")
plt.show()

#%%
# Hierarchical Tree
clusterer.condensed_tree_.plot()
plt.show()
clusterer.single_linkage_tree_.plot()
plt.show()

#%%
hierarchy = clusterer.cluster_hierarchy_
alt_labels = hierarchy.get_clusters(0.100, 5)
hierarchy.plot()
# %% FCM
from fcmeans import FCM

fcm = FCM(n_clusters=10)
fcm.fit(df.to_numpy())
fcm_centers = fcm.centers
fcm_labels = fcm.predict(df.to_numpy())
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in fcm_labels]
plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
plt.title("FCM")
plt.show()

