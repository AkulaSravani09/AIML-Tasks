#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


movies_df = pd.read_csv("Movie.csv")
movies_df


# In[11]:


# verify that a single userid has rated more than once
movies_df[movies_df["userId"]==11]


# In[13]:


movies_df.info()


# In[23]:


# plot bar chart for number of times each movie is rated
counts = movies_df['movie'].value_counts().reset_index()
print(counts)
 #columns =["movie", "counts"]
plt.xlabel("Movies")
plt.ylabel("Counts")
plt.xticks(rotation=45, ha='right')
sns.barplot(data=counts,x="movie", y='count',hue ='movie', palette="Set2")


# In[25]:


# print the number of unique users

print(movies_df.userId.unique())
len(movies_df.userId.unique())


# In[29]:


# sort the dataframe in as per user id
movies_df.sort_values('userId')


# In[31]:


# print bar chart for counting the ratings category-wise

counts =  movies_df['rating'].value_counts().reset_index()
print(counts)
plt.xlabel("Ratings")
plt.ylabel("Counts")
sns.barplot(data=counts,x='rating', hue= 'rating', y = 'count', palette="Set2")


# ### Transform the table

# In[34]:


user_movies_df = movies_df.pivot_table(index='userId',columns='movie',values='rating')


# In[36]:


user_movies_df


# In[38]:


# impute those NaNs with 0 values
user_movies_df.fillna(0, inplace=True)


# In[40]:


user_movies_df


# In[ ]:


## Generate similar users data


# In[42]:


# calculating cosine similarity between users
from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import cosine, correlation


# In[44]:


user_sim = 1 - pairwise_distances(user_movies_df.values,metric='cosine')


# In[46]:


user_sim


# In[48]:


user_sim.shape


# In[52]:


np.fill_diagonal(user_sim, 0)
user_sim


# In[54]:


# store the results in a dataframe
user_sim_df=pd.DataFrame(user_sim)
user_sim_df


# In[56]:


movies_df.userId.unique()


# In[60]:


# set the index an dcolumn names to user ids
user_sim_df.index=movies_df.userId.unique()
user_sim_df.columns=movies_df.userId.unique()


# In[62]:


user_sim_df


# In[64]:


# most Similar Users
user_sim_df.idxmax(axis=1)[0:50]


# In[ ]:




