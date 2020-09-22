import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df=pd.read_csv("crime_data.csv", index_col=0)
# print(df)
# print(df.head(5))
# input()
# print(df.describe())
# print(df.info())
# input()

from sklearn import preprocessing
crime_rates_standardized=preprocessing.scale(df)
# print(crime_rates_standardized)
crime_rates_standardized = pd.DataFrame(crime_rates_standardized)

from sklearn.cluster import KMeans

plt.figure(figsize=(10, 8))
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(crime_rates_standardized)
    wcss.append(kmeans.inertia_)    #criterion based on which K-means clustering works
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
# input()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(crime_rates_standardized)

# print(np.array(y_kmeans))
# input()

y_kmeans1=y_kmeans+1
# print(y_kmeans1)
# input()

# New list called cluster
cluster = list(y_kmeans1)
# Adding cluster to this data set
df['cluster'] = cluster
print(df.head(10))
input()

kmeans_mean_cluster=pd.DataFrame(round(df.groupby('cluster').mean(),1))
print(kmeans_mean_cluster)

plt.figure(figsize=(12,6))
sb.scatterplot(x=df['Murder'], y = df['Assault'],hue=y_kmeans1)
plt.show()

print(df[df['cluster']==1])
