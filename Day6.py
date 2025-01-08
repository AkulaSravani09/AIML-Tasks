#!/usr/bin/env python
# coding: utf-8

# In[1]:


num = 6
if num %2 == 0:
    print("even")
else:
    print("odd")


# In[3]:


x = 10
result = "positive" if x > 0 else "negative"
print(result)


# In[5]:


age = 18
category = "Adult" if age >= 18 else "Minor"
print(category)


# In[7]:


#nested if-else
num = 0
result = "Positive" if num > 0 else "Negative" if num < 0 else "Zero" 
print(result)


# In[9]:


#nested if-else
num = 2
result = "Positive" if num > 0 else "Negative" if num < 0 else "Zero" 
print(result)


# In[11]:


#nested if-else
num = -5
result = "Positive" if num > 0 else "Negative" if num < 0 else "Zero" 
print(result)


# In[13]:


#List Comprehension
L = [1,9,2,10,56,89]
[2*x for x in L]


# In[15]:


#List Comprehension any alphabet can keep
L = [1,9,2,10,56,89]
[2*a for a in L]


# In[17]:


#print even numbers
L = [1,9,2,10,56,89]
[x for x in L if x%2 == 0]


# In[19]:


#print odd numbers
L = [1,9,2,10,56,89]
[x for x in L if x%2 != 0]


# In[23]:


#dictionary comprehension
d1 = {'Sravani':[80, 85, 79, 90], 'Asha':[90,70,85,82], 'Laddu':[90,60,55,50], 'Akki':[49,99,86,74]}
d1


# In[25]:


{k:sum(v)/len(v) 
 for k,v in d1.items()}


# In[27]:


for name, scores in d1.items():
    avg = sum(scores) / len(scores)
    print(f"{name}: {avg}")


# In[33]:


#Define function (reusable)
def mean_value(given_list):
    total = sum(given_list)
    average_value = total/len(given_list)
    return average_value


# In[35]:


#Call the function
L = [1,2,3,4,5,6,7,8,9,10]
mean_value(L)


# In[37]:


L = [1,2,3,4,5,6,7,8,9,10]
total = sum(L)
total


# In[39]:


def greet(name):
    return f"Good Morning, {name}!"
# Call the function with your name
message = greet("Sravani")
print(message)


# In[41]:


def greet(name):
    print(f"Good morning, {name}!")

greet("Akki")


# In[ ]:




