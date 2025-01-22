#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np


# In[14]:


df = pd.read_csv("Universities.csv")
df


# In[16]:


#mean value of sat score
np.mean(df["SAT"])


# In[18]:


np.median(df["SAT"])


# In[20]:


#standard deviation of data
np.std(df["GradRate"])


# In[22]:


#find the variance 
np.var(df["SFRatio"])


# In[24]:


df.describe()


# In[ ]:


#name of people,places here we use nominal data type,gender-nominal datatype/categorical data, continuous/ratio data means which can have a decimal type
#categorical data we use bar graph visualization
#continuous fractional values like height,weight of student, stock price here we use histogram visualization


# In[ ]:


#visualizations


# In[26]:


#visulaise the GradRate using histogram
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[ ]:




