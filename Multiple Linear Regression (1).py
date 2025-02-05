#!/usr/bin/env python
# coding: utf-8

# #### Assumptions in multilinear regression
# 1.**Linearity:** The relationship between the predictors(X) and the response (Y) is linear.
# 
# 2.**Independence:** Observations are independent of each other.
# 
# 3.**Homoscedasticity:** The residuals (Y - Y_hat) exhibit constant varinace at all levels of the predictor.
# 
# 4.**Normal Distribution Of Errors:** The residuals of the model are normally distributed.
# 
# 5.**No multicollinearity:** The independent variable should not be too highly correlated with each other.
# 
# violations of these assumptions may lead to inefficiency in the regression paramaeyers and unreliable predictions.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


#Read the data csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


#rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ## Descriptiom of columns
# - MPG: Milege of the car  (Miles Per Gallon) (This is Y-column to be predicted)
# - Hp:Horsepower of car (X1 column)
# - VOL: Volume of the car (size) (x2 column)
# - WT: Weight of the car (pounds) (x3 column)
# - SP: Top speed of car (Miles per hour) (x4 column)
# - x1,x2,x3,x4 are called as **independent varibale**
# - another name of **indenpent varible is features of the dataset and predictors**
# - y is called as **dependent variable**
# - Model eqn:**Y=Y^+error**

# ##### EDA

# In[20]:


cars.info()


# In[22]:


#check for missing values
cars.isna().sum()


# ### observations 
# - There are no missing values
# - There are 81 observations (81 different cars data)
# - The data types of the columns are also relevnat and valid

# In[28]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') #remove x label for the boxplot

#creating a histgram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#adjust layout
plt.tight_layout()
plt.show()


# In[30]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='') #remove x label for the boxplot

#creating a histgram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#adjust layout
plt.tight_layout()
plt.show()


# In[32]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='') #remove x label for the boxplot

#creating a histgram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#adjust layout
plt.tight_layout()
plt.show()


# In[34]:


#create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

#creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='') #remove x label for the boxplot

#creating a histgram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#adjust layout
plt.tight_layout()
plt.show()


# ##### observations from boxplot and histogram
# - There are some extreme values(outliers) observed in towrads the right tail of SP and HP distributions.
# - In VOL and WTcolumns, a few outliers are observed in both tails of their distributions.
# - The extreme values of cars data may have come from the specially designed natire of cars
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may havee to be considered while buildig the regression model

# #### checking for duplicated rows

# In[38]:


cars[cars.duplicated()]


# ## pair plots and correlation coefficients

# In[42]:


#pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[44]:


cars.corr()


# - highest between wt and vol
# - second highest is between hp and sp
