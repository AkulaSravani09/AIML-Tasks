#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# data1.info()

# In[4]:


data1.describe()


# In[22]:


data1.isnull().sum()


# ### Correlation

# In[31]:


data1["daily"].corr(data1["sunday"])


# In[33]:


data1[["daily","sunday"]].corr()


# In[35]:


data1.corr(numeric_only=True)


# In[39]:


plt.scatter(data1["daily"], data1["sunday"])
plt.show()


# In[45]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[57]:


plt.show()


# In[51]:


plt.scatter(data1["daily"], data1["sunday"])
plt.show()


# In[59]:


sns.histplot(data1["daily"], kde=True)
plt.show()


# In[ ]:




