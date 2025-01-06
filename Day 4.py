#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Dictionary
#create dictionary using curly braces{}
d1 = {"a": 1, "b": 2, "c": 3}
print(d1)


# In[5]:


#create dictinoary with dict() constructor
d2 = dict(A=10,B=20,C=30)
print(d2)


# In[9]:


#create dictionary with a list of tuples
d3 = dict([("x", 1),("y", 2), ("z", 3)])
print(d3)


# In[11]:


d1


# In[13]:


#print keys,value and key:value pairs
print(d1.keys())
print(d1.values())
print(d1.items())


# In[15]:


dir(dict)


# In[31]:


scores = {"virat":90,"Rohit":100,"Rahul":70,"Hardik":9,"Gill":20,"Jadeja":30,"Siraz":15}
scores


# In[33]:


#adding 
scores["Aswin"]=15
scores


# In[35]:


# update jadeja score to 50 using update() method
scores.update({"Jadeja":50})
scores


# In[45]:


#update() method modifies the existing keys and acn add new key:value pair
scores.update({"Hardik":25,'Rahane':55})
scores


# In[47]:


#update can take more than two values but here setdefault takes only 1 value
#it ignores the key value if the key is already present
#it will add if the key is not present
scores.setdefault("Rahul",60)
scores


# In[55]:


scores.setdefault("Pujara",50)
scores


# In[59]:


#removing the elements form dictionary
#1.pop(key) removes the particular element
#2.popitem() removes the last element
scores.popitem()
scores


# In[61]:


scores.popitem()
scores


# In[63]:


scores.popitem()
scores


# In[65]:


scores.pop("Rahane")
scores


# In[69]:


scores.get("Siraz")


# In[85]:


#use of fromkeys() to create a new dict
list1=["A","B","C","D"]
my_dict=dict.fromkeys(list1)
my_dict


# In[83]:


#use of fromkeys() to create a new dict
list1=["A","B","C","D"]
my_dict=dict.fromkeys(list1,10)
my_dict


# In[77]:


my_dict.update({"A":10,"B":20})
my_dict


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




