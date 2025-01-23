#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[5]:


#mean value of sat score
np.mean(df["SAT"])


# In[7]:


np.median(df["SAT"])


# In[9]:


#standard deviation of data
np.std(df["GradRate"])


# In[11]:


#find the variance 
np.var(df["SFRatio"])


# In[13]:


df.describe()


# In[ ]:


#name of people,places here we use nominal data type,gender-nominal datatype/categorical data, continuous/ratio data means which can have a decimal type
#categorical data we use bar graph visualization
#continuous fractional values like height,weight of student, stock price here we use histogram visualization


# In[ ]:


#visualizations


# In[15]:


#visulaise the GradRate using histogram
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[28]:


#visualization using boxplot
s = [20,15,10,25,30,35,28,40,45,60]
scores = pd.Series(s)
scores


# In[30]:


plt.boxplot(scores, vert = False)


# In[32]:


s = [20,15,10,25,30,35,28,40,45,60,120,150]
scores = pd.Series(s)
scores


# In[34]:


plt.boxplot(scores, vert = False)


# In[38]:


#visualization from universities dataset to identify the outliers
df = pd.read_csv("Universities.csv")
df


# In[56]:


plt.figure(figsize=(6,2))
plt.title("Box plot for SAT score")
plt.boxplot(df["SAT"],vert=False)


# In[60]:


plt.figure(figsize=(6,2))
plt.title("Box plot for Top10")
plt.boxplot(df["Top10"],vert=False)


# In[62]:


plt.figure(figsize=(6,2))
plt.title("Box plot for Accept")
plt.boxplot(df["Accept"],vert=False)


# In[64]:


plt.figure(figsize=(6,2))
plt.title("Box plot for SFRatio")
plt.boxplot(df["SFRatio"],vert=False)


# In[ ]:





# In[ ]:




