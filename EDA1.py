#!/usr/bin/env python
# coding: utf-8

# In[7]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[ ]:


# in the above table y is the last column i.e called as target column or y column remaining all the columns arre the x columns


# In[15]:


#printing the information about the table
data.info()


# In[ ]:


#total are 158 non null values so in the frst ozone it has 38 nan values,2nd one has 7 nan values and so on...........


# In[19]:


#Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[21]:


#drop duplicate column(temp c)and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis =1)
data1


# In[25]:


data1.info()


# In[29]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[33]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[37]:


#drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[ ]:





# In[ ]:




