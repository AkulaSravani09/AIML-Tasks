#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np


# In[9]:


df = pd.read_csv("universities.csv")
df


# In[11]:


df.sort_values(by="GradRate",ascending=True)


# In[16]:


df[(df["GradRate"]>=95)]


# In[18]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[ ]:





# In[ ]:




