
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
import seaborn as sns
import numpy as np



#%%
path = r'Concrete_Data_Yeh.csv'
df = pd.read_csv(path)
df.dropna(inplace=True)
cat_columns = df.select_dtypes(['object']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
projection = TSNE().fit_transform(df)
sns.scatterplot(*projection.T, s=50, linewidth=0, hue=df.csMPa, alpha=0.25)
plt.title("Category")
plt.show()

#%%
clusterer = hdbscan.HDBSCAN(min_cluster_size=10).fit(df)
sns.scatterplot(*projection.T, s=50, linewidth=0, hue=(clusterer.labels_-8)*-1, alpha=0.25)
plt.title("HDBSCAN")
plt.show()
uni,cou = np.unique(clusterer.labels_,return_counts=True)
#%%
from fcmeans import FCM

fcm = FCM(n_clusters=10)
fcm.fit(df.to_numpy())
fcm_centers = fcm.centers
fcm_labels = fcm.predict(df.to_numpy())
sns.scatterplot(*projection.T, s=50, linewidth=0, hue=fcm_labels, alpha=0.25)
plt.title("FCM")
plt.show()