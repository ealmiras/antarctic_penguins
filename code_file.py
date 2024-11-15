import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv('penguins.csv')
print(penguins_df.head())

p_df = pd.get_dummies(penguins_df, dtype='int')
p_df.head()

scaler = StandardScaler()
X = scaler.fit_transform(p_df)
p_scaled = pd.DataFrame(X, columns = p_df.columns)
p_scaled.head()

inertia = []
for k in range(1,10):
    kmeans = KMeans(n_clusters=k).fit(p_scaled)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters).fit(p_scaled)
p_df['label'] = kmeans.labels_

plt.scatter(p_df['label'], p_df['culmen_length_mm'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Cluster')
plt.ylabel('culmen_length_mm')
plt.xticks(range(int(p_df['label'].min()), int(p_df['label'].max()) + 1))
plt.title(f'K-means Clustering (K={n_clusters})')
plt.show()

numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','label']
stat_penguins = p_df[numeric_columns].groupby('label').mean()
stat_penguins