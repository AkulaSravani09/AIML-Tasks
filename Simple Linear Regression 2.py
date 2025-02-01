#!/usr/bin/env python
# coding: utf-8

# #### Import datasets and libraries

# In[13]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[15]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# #### Linear regression
# - method of obtaining a straight line eqn btw x and y 
# - x means independent variable
# - y means dependent variable
# - if y column is having a numerical float values i.e. continous values then it is regression
# - if it has categorical value then it is simplification
# - if it has multiple x values and  a single y column then it is called multiple linear regression

# In[17]:


data1.info()


# In[23]:


data1.isnull().sum()


# In[21]:


data1.describe()


# In[25]:


#Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert=False)
plt.show()


# In[27]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[29]:


plt.figure(figsize=(6,3))
plt.title("Box plot for sunday")
plt.boxplot(data1["sunday"], vert=False)
plt.show()


# In[31]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# ### observations
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday Column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the 
# 

# ## Scatter plot and correlation strength

# In[39]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show


# In[44]:


data1["daily"].corr(data1["sunday"])


# In[46]:


data1[["daily","sunday"]].corr()


# In[48]:


data1.corr(numeric_only=True)


# ## Observations on correlation strength
# - The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong and positve with Pearson's correlation coefficient of 0.958154

# ## Fit a Linear Regression Model

# In[59]:


# build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[61]:


model1.summary()


# In[ ]:




