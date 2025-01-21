#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Universities.csv")
df


# In[5]:


#mean value of sat score
np.mean(df["SAT"])


# In[9]:


np.median(df["SAT"])


# In[ ]:


#standard deviation of data
np.std


# In[11]:


#find the variance 
np.var(df["SFRatio"])


# In[13]:


df.describe()


# In[ ]:





# In[ ]:





# In[ ]:




