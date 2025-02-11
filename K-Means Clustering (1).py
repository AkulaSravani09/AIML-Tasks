#!/usr/bin/env python
# coding: utf-8

# - There are only x columns no target columns (y) in unsupervised columns
# - Clustering is one of the example of this unsupervised learning
# - Clustering means divide the records into groups (Clusters)

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### clustering-DIvide tghe universities into groups(clusters)

# In[4]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[5]:


Univ.info()


# In[6]:


Univ.isnull().sum()


# In[7]:


Univ.boxplot()


# In[8]:


Univ.hist()


# In[9]:


Univ.describe()


# In[10]:


plt.figure(figsize=(6,2))
plt.boxplot(Univ["SAT"], vert= False)


# In[11]:


plt.show()


# ## Standardization of the data

# In[12]:


# read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[13]:


Univ1


# In[14]:


cols = Univ1.columns


# In[15]:


#Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns = cols)
scaled_Univ_df
#scaler.fit_transform(Univ1)


# In[33]:


# Build 3 clusters using KMeans Cluster Algorithm\
from sklearn.cluster import  KMeans
clusters_new = KMeans(3, random_state=0) # Specify # clusters
clusters_new.fit(scaled_Univ_df)


# In[35]:


#print the cluster labels
clusters_new.labels_


# In[37]:


set(clusters_new.labels_)


# In[41]:


#Assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_
Univ


# In[43]:


Univ.sort_values(by = "clusterid_new")


# In[47]:


# Use groupby () to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations
# - Cluster 2 appears to be the top rated universities cluster as the cut off score, Top10, SFRatio parameter mean values are highest
# - Cluster 1 appears to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities

# ## Finding optimal K value using elbow point

# In[55]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
    #kmeans.fit(Univ1)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# ### Observations
# - From the above graph we can choose K = 3 or 4 which indicates the elbow joint that is the rate of change of slope decreases

# ### clustering methods
# 1. Hierarchical clustering
# 2. K Means Clustering
# 3. K Medoids clustering
# 4. K Prototypes clustering
# 5. DBSCAN
