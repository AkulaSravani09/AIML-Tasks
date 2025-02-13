#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install mlxtend library
#!pip install mlxtend


# In[ ]:


# from mlxtend.preprocessing import TransactionEncoder


# In[5]:


# Import necessary libraries

import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[7]:


#print the dataframe
titanic = pd.read_csv("Titanic.csv")
titanic


# In[9]:


titanic.info()


# In[11]:


titanic.describe()


# In[15]:


titanic.isnull().sum()


# ### Observations
# - There are no null values
# - All columns are object data type and categorical in nature
# - As the columns are categorical, we can adopt one-hot-encoding
# 

# In[37]:


#plot a bar chart to visualise the category of people on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[33]:


plt.show()


# ##### OBSERVATIONS
# - Maximum travelers are crew
# - next 3rd class
# - then 1st class and at last 2nd class

# In[55]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[57]:


plt.show()


# - highest are adult

# In[44]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[46]:


plt.show()


# - Male are more

# In[48]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[50]:


plt.show()


# In[59]:


# perform onehot encoding on categorical columns
df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[61]:


df.info()


# ### Apriori Algorithm

# In[68]:


#Apply Apriori algorithm to get itemset combinations
frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[70]:


frequent_itemsets.info()


# In[74]:


#Generate association rules with metrics
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# - lift value > 1 then there is some association
# - if lift is more then the strong is asscoiation

# In[78]:


rules.sort_values(by='lift', ascending = True)


# In[82]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()

