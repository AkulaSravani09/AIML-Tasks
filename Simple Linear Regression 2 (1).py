#!/usr/bin/env python
# coding: utf-8

# #### Import datasets and libraries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# #### Linear regression
# - method of obtaining a straight line eqn btw x and y 
# - x means independent variable
# - y means dependent variable
# - if y column is having a numerical float values i.e. continous values then it is regression
# - if it has categorical value then it is simplification
# - if it has multiple x values and  a single y column then it is called multiple linear regression

# In[5]:


data1.info()


# In[6]:


data1.isnull().sum()


# In[7]:


data1.describe()


# In[8]:


#Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert=False)
plt.show()


# In[9]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[10]:


plt.figure(figsize=(6,3))
plt.title("Box plot for sunday")
plt.boxplot(data1["sunday"], vert=False)
plt.show()


# In[11]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# ### observations
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday Column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the 
# 

# ## Scatter plot and correlation strength

# In[14]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show


# In[15]:


data1["daily"].corr(data1["sunday"])


# In[16]:


data1[["daily","sunday"]].corr()


# In[17]:


data1.corr(numeric_only=True)


# ## Observations on correlation strength
# - The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong and positve with Pearson's correlation coefficient of 0.958154

# ## Fit a Linear Regression Model

# In[20]:


# build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[23]:


model1.summary()


#  # Model Summary
#  - How many data points are closer to the fitted line is described by R-Squared
#  - R - squared is called co-efficient of determination
#  - It lies between 0 to 1 i.e; 1 means excellent
#  - 91 % varience in y is explained by x
#  - p values : are the beta_0, beta_1 significant?
#  - if p < 0.05 significant
#  - if p value is lessthan 0.05 then it is significant otherwise not significant
# # Observations
# - The probability (p-value) for intercept(beta_0) is 0.707>0.05
# - Therefore the intercept coeeficinet may not be that much significant in prediction
# - However the p-value for "daily" (beta_1) is 0.00 < 0.05
# - Therefore the beta_1 coefficient is highly significant and is contributint to prediction

# # Interpretation:
# - R squared = 1 perfect  fit(all variance explained)
# - R squared = 0 Model does not explain any variance
# - R squared close to 1 Good model fit
# - R squared close to 0 Poor model fit

# In[44]:


# plot the scatter plot and overlay the fitted st line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
#predicteed response vector
y_hat = b0 + b1*x
#plotting the regression line
plt.plot(x, y_hat, color = "g")
#putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[51]:


model1.params


# In[55]:


#print the model statistice( t and p-values)
print(f'model t-values:\n{model1.tvalues}\n--------------\nmodel p-values: \n{model1.pvalues}')


# In[59]:


#print the Quality of fitted line(R-squared values
(model1.rsquared,model1.rsquared_adj)


# ### Predict for new data point

# In[63]:


#predict for 200 and 300 daily circulation
newdata=pd.Series([200,300,1500])


# In[65]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[67]:


model1.predict(data_pred)


# ##### y - y^ = error
# - performance metric is
# - 1/n[e1 square+e2 square+-----------en square]=Mean Square Error

# In[73]:


#predicate on all given training darta
pred = model1.predict(data1["daily"])


# In[75]:


#Add predicted values as a column in detail
data1["Y_hat"] = pred
data1


# In[77]:


#compute the error values (residulas) and add as another column
data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1


# In[81]:


#compute mean squared error for the model
mse=np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE:  ",mse)
print("RMSE: ",rmse)

