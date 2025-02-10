#!/usr/bin/env python
# coding: utf-8

# - There are only x columns no target columns (y) in unsupervised columns
# - Clustering is one of the example of this unsupervised learning
# - Clustering means divide the records into groups (Clusters)

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ### clustering-DIvide tghe universities into groups(clusters)

# In[7]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[12]:


Univ.info()


# In[14]:


Univ.isnull().sum()


# In[16]:


Univ.boxplot()


# In[18]:


Univ.hist()


# In[20]:


Univ.describe()


# In[34]:


plt.figure(figsize=(6,2))
plt.boxplot(Univ["SAT"], vert= False)


# In[36]:


plt.show()


# In[42]:


# read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]


# In[44]:


Univ1


# In[56]:


cols = Univ1.columns


# In[52]:


#Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns = cols)
scaled_Univ_df
#scaler.fit_transform(Univ1)


# In[ ]:




