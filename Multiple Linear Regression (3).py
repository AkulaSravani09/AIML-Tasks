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

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


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
# -** Y^ = Beta0+beta1X1+beta2X2+beta3X3+beta4X4** : eqn
# - beta0,beta1,beta2,beta3,beta4 are called as **model coefficents**
# - row represnts different cars data
# - **81 cars**
# - less than 0.5 then some of the values are valid

# ##### EDA

# In[7]:


cars.info()


# In[8]:


#check for missing values
cars.isna().sum()


# ### observations 
# - There are no missing values
# - There are 81 observations (81 different cars data)
# - The data types of the columns are also relevnat and valid

# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


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

# In[16]:


cars[cars.duplicated()]


# ## pair plots and correlation coefficients

# In[18]:


#pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[46]:


cars.corr()


# ###  observations from correlation plots and coefficients
# - Between x and y all the x avribales are showing moderate to high correlation strengths,highest being between HP and MPG
# - Therfore this dataset qualifiess for building a multiple linear regression model to predict MPG
# - Among x columns (x1,x2,x3,x4),x+some very high correlation strengths are observed between SP vs HP, VOL vs Wt
# - The high correlation among x columns i snot desirable as it might lead to multicollineraity problme.

#  #### preparing a preliminary model considering alll x columns

# In[22]:


#build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()


# In[23]:


model1.summary()


# - r squareed values tells that how many points are closer to the line 
# - max values of r is 1
# - min values is 0
# - beta0 is **intercept coefficient**, beta1 is **wt coeffiecient** like beta 2 beta3 and beta4 are same like the above
# - y^ = 30.677+0.4006(WT)-0.3361(VOL)+0.3956(SP)-0.2054(HP) model eqn observed from the model summary
# - the **p>|t| values should be less than 0.5** here WT AND VOL are more than that
# - first we look at **r squared vales and adj r sqaured values then f_statistic** and so on

# ### observations from model summary
# - The R-squared and adjusted R-squared values are good and about 75% of variablity in Y is explained by X xolumns
# - The probability value with respect to F-statistic is close to zero, indicating that all or someof x columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored

# ### performance metrucs for model1

# In[27]:


# Find the performance metrics
# Create a data frame with actual y and predicted y columns
df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# - perfomance mertric we have used is mean square error
# - to fing y_hat values we used predict method
# - y - y^ = ERROR
# -  1/n[e1 square+e2 square+-----------en square]=Mean Square Error
# -  RMSE is square root of MSE

# In[29]:


# predict for the given x data columns
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[30]:


# compute the Mean Squared Error(MSE), RSME for model1

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### checking for multicollinearity among X- columns using VIF method

# In[ ]:


#cars.head()


# In[54]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ### observations for VIF values:
# - The ideal range of VIF values shall be between 0 to 10.However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOl and Wt, it is clear that they are prone to multicollinearity proble.
# - Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicollinearity
# - It is decided to drop WT and retain VOL column in further models.

# In[57]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[61]:


model2 = smf.ols('MPG~HP+VOL+SP', data=cars1).fit()
model2.summary()


# ### perfomance metrics for model2

# In[66]:


# find the perfomance metrics
#  create a data frame with actual y nad predicted y columns

df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[70]:


# Predict for the given X data columns
pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[72]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### observations from model2 summary()
# - The adjusted R-squared value imporved slightly to 0.76
# - All the p-values for model parameters are less than 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalised as the significant predictor for the MPG response variable
# - There is no improvement in MSE value
