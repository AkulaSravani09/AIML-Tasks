#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[ ]:


# in the above table y is the last column i.e called as target column or y column remaining all the columns arre the x columns


# In[3]:


#printing the information about the table
data.info()


# In[ ]:


#total are 158 non null values so in the frst ozone it has 38 nan values,2nd one has 7 nan values and so on...........


# In[7]:


#Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[9]:


#drop duplicate column(temp c)and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis =1)
data1


# In[11]:


data1.info()


# In[13]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[15]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[17]:


#drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[ ]:


####RENAME THE COLUMNS


# In[21]:


#change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[ ]:


####Impute the missing values


# In[23]:


#Display data1 info()
data1.info


# In[27]:


#HANDLING MISSING VALUES IN THE TABLE
#display data1 missing value count in each column using isnull().sum()
data1.isnull().sum()


# In[37]:


#visualize data1 missimg values using heat map
cols = data1.columns
colors = ['black', 'white']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[ ]:


#we cant delete the complete rows if there are missing values 
#we have some methodologies to replace the missing values 
#we can replace it with median or mean value in the table


# In[ ]:


#categorical means it have categories like ps,c,s in weather and month cant be 5.5 or something month can be 8 whuch means august,day also categorical data only
#in this table categorical data areweather,month,day  
#replace the categorical value with the mode 


# In[39]:


#find the mean and median values of each numeric 
#imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[43]:


#replace the ozen missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[51]:


mean_solar = data1["Solar"].mean()
median_solar = data1["Solar"].median()
print("Mean Solar: ", mean_solar)
print("Median of Solar: ", median_solar)


# In[55]:


data1['Solar'] = data1['Solar'].fillna(mean_solar)
data1.isnull().sum()


# In[ ]:




