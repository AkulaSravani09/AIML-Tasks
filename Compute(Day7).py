#!/usr/bin/env python
# coding: utf-8

# In[13]:


#function with variable number of arguments
def mean_value(*n):
    sum = 0
    counter = 0
    for x in n:
        counter = counter +1
        sum += x
    mean = sum / counter
    return mean


# In[17]:


def median_value(*n):
    num_list = list(n)
    num_list.sort()
    l = len(num_list)
    if l%2 == 0:
        median = (num_list[int(l/2)] + num_list[int((l/2))-1])/2
    else:
        median = num_list[int(l/2)]
    return median


# In[21]:


mean_value(2,4,6,8,9,12,16,24)


# In[23]:


median_value(2,4,6,8,9,12,16,24)

