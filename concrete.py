
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
