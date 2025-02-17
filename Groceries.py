#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


groceries_df = pd.read_csv("Groceries_dataset.csv")
groceries_df


# In[15]:


groceries_df.head()


# In[17]:


groceries_df.tail()


# In[19]:


groceries_df.info()


# In[21]:


groceries_df.describe()


# In[23]:


groceries_df.isnull().sum()


# In[34]:


groceries_df.duplicated()


# In[26]:


counts = groceries_df['Member_number'].value_counts()
plt.bar(counts.index, counts.values)


# In[28]:


counts = groceries_df['Date'].value_counts()
plt.bar(counts.index, counts.values)


# In[30]:


counts = groceries_df['itemDescription'].value_counts()
plt.bar(counts.index, counts.values)


# In[38]:


get_ipython().system('pip install mlxtend')


# In[44]:


print(groceries_df.Member_number.unique())
len(groceries_df.Member_number.unique())


# In[46]:


groceries_df.sort_values('Member_number')


# In[62]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# In[60]:


# Example dataset: Load transactional data (Ensure your dataset is in the right format)
df = pd.read_csv("groceries_dataset.csv")  # Replace with your CSV file

# Ensure all columns are Boolean (0s and 1s) for apriori
df_encoded = df.astype(bool)  # Convert numerical data to True/False


# In[56]:


# Find frequent itemsets with a minimum support threshold
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
print(frequent_itemsets)


# In[58]:


# Generate association rules with confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules)


# In[64]:


# Assuming you already have a DataFrame loaded as 'df'
# Checking for null values
null_values = df.isnull().sum()

# Checking for duplicate rows
duplicates = df.duplicated().sum()

# Print the results
print("Null values per column:\n", null_values)
print("\nTotal duplicate rows: ", duplicates)


# In[86]:


cols = groceries_df.columns
colors = ['black', 'white']
sns.heatmap(groceries_df[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)

