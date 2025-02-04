#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ## Descriptiom of columns
# - Hp:Horsepower of engine
# - MPG: Milege Miles Per Gallon
# - VOL: Volume
# - WT: Weight
# - SP: Top speed of car
# - x1,x2,x3,x4 are called as

# In[7]:


#rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()

