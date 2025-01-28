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


# In[3]:


# in the above table y is the last column i.e called as target column or y column remaining all the columns arre the x columns


# In[4]:


#printing the information about the table
data.info()


# In[5]:


#total are 158 non null values so in the frst ozone it has 38 nan values,2nd one has 7 nan values and so on...........


# In[6]:


#Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[7]:


#drop duplicate column(temp c)and unnamed column
data1 = data.drop(['Unnamed: 0', "Temp C"], axis =1)
data1


# In[8]:


data1.info()


# In[9]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[10]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[11]:


#drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[12]:


####RENAME THE COLUMNS


# In[13]:


#change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[14]:


####Impute the missing values


# In[15]:


#Display data1 info()
data1.info


# In[16]:


#HANDLING MISSING VALUES IN THE TABLE
#display data1 missing value count in each column using isnull().sum()
data1.isnull().sum()


# In[17]:


#visualize data1 missimg values using heat map
cols = data1.columns
colors = ['black', 'white']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[18]:


#we cant delete the complete rows if there are missing values 
#we have some methodologies to replace the missing values 
#we can replace it with median or mean value in the table


# In[19]:


#categorical means it have categories like ps,c,s in weather and month cant be 5.5 or something month can be 8 whuch means august,day also categorical data only
#in this table categorical data areweather,month,day  
#replace the categorical value with the mode 


# In[20]:


#find the mean and median values of each numeric 
#imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[21]:


#replace the ozen missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[22]:


mean_solar = data1["Solar"].mean()
median_solar = data1["Solar"].median()
print("Mean Solar: ", mean_solar)
print("Median of Solar: ", median_solar)


# In[23]:


data1['Solar'] = data1['Solar'].fillna(mean_solar)
data1.isnull().sum()


# In[24]:


#find the mode values of categorical column(weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[25]:


#impute missing values (Replace NaN with mode etc.) of "weather" using fillna
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[26]:


print(data1["Month"].value_counts())
mode_Month = data1["Month"].mode()[0]
print(mode_Month)


# In[27]:


data1["Month"] = data1["Month"].fillna(mode_Month)
data1.isnull().sum()


# In[28]:


#outliers in dataset are some are having high values compared to remaining values
#bocplot and histogram are 2 popular visulaizations to find the outliers


# In[29]:


#detection of outliers in the columns
#method1: using histogram and boxplots
#create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})
 #plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

#plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")

#adjust layout for better spacing
plt.tight_layout()                        

#show the plot
plt.show()                       


# In[30]:


###Observations
#The ozone column has extreme values beyond 81 as seen from box plot
# The same is confirmed from the below right_skewed histogram


# In[31]:


fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios':[1,3]})
 #plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Solar"], ax=axes[0], color='blue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

#plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='black', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")

#adjust layout for better spacing
plt.tight_layout()                        

#show the plot
plt.show()      


# In[32]:


#observations
#here no ouliers are observed in the solar from boxplot
#left_skewed histogram


# In[33]:


#create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='purple')
#show the plot
plt.show()


# In[34]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert= False)


# In[70]:


#extract oyliers from the boxlpot for ozone column
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert = False)
[item.get_xdata() for item in boxplot_data['fliers']]
#fliers are outliers


# In[72]:


#method 2 
#using mu +/- 3* sigma limits(standard deviation method)
data1["Ozone"].describe()


# In[84]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[ ]:


##observations
#it is observed that only two outliers are identified
#in box plot method more number of outliers are observed
#this is beacuse the assumption of normality is not satisfied in this column


# In[94]:


#quantile-quantile plot fro detection of outliers
import scipy.stats as stats
#create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot fro outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[ ]:


#observation sfrom Q-Q plot
#the data does not follow normal distribution as the data points are deviating significantly away from the red line
#the data shows a right-skewed distribution and possbile outliers


# In[96]:


import scipy.stats as stats
#create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(data1["Solar"], dist="norm", plot=plt)
plt.title("Q-Q plot fro outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[ ]:


#other visualizations that could help in the ddetection of outliers
#create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.tile("Violin Plot")

