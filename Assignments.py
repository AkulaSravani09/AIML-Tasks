#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Day1
a = 3
b = 4.5
c = "hi"
d = True
print(a)
print(type(a))
print(b)
print(type(b))
print(c)
print(type(c))
print(d)
print(type(d))


# In[19]:


#Day 2
L = [1,2,3,4,5]
Tup = 1,2,3,4,5
dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
print(L)
print(Tup)
print(dict)
print(L[1:3])
print(Tup[1:3])
print(dict['a'])



# In[35]:


#Day 3
marks1 = float(input("Enter marks for sub1: "))
marks2 = float(input("Enter marks for sub2: "))
marks3 = float(input("Enter marks for sub3: "))
a = (marks1 + marks2 + marks3) / 3
if a >= 90:
    print("Grade A")
elif a >= 89:
    print("Grade B")
elif a>= 79:
    print("Grade c")


# In[ ]:




