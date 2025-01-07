#!/usr/bin/env python
# coding: utf-8

# In[1]:


#sets
#unordered
#mutable
#unique values#allows immutable elements
#no indexing or slicing
#creating a set with curly braces
s1= {1,8,9,0,10,9,0,1,9}
print(s1)
print(type(s1))


# In[5]:


lst1 = [1,8,9,0,10,20,78,8,8,8]


# In[9]:


s2 = set(lst1)
print(s2)
print(type(s2))


# In[11]:


dir(set)


# In[15]:


#union operation using | operator 
s1 = {1,2,3,4,5} 
s2 = {3,4,5,6}
s1 | s2



# In[17]:


s1.union(s2)


# In[19]:


#instersection using & operator
s1 = {1,2,3,4}
s2 =  {3,4,5,6}
s1 & s2


# In[21]:


s1.intersection(s2)


# In[23]:


#difference of  two sets
s1 = { 2,3,5,6,7}
s2 = {5,6,7}
s1 - s2


# In[27]:


# s1 - s2 != s2 - s1
s2 - s1


# In[29]:


#Symmetric difference
s1={1,2,3,4,5}
s2={4,5,6,7,8}
s1.symmetric_difference(s2)


# In[31]:


s2.symmetric_difference(s1)


# In[41]:


#strings
str1 = "Welcome to aiml class"
print(str1)
str2 = 'We started with python'
print(str2)
str3 = '''This is an awesome class'''
print(str3)
print(type(str1))
print(type(str2))
print(type(str3))


# In[49]:


str4 = '''He said, "It's is awesome!"'''
print(str4)


# In[55]:


#slicing in strings
print(str1)
str1[5:10]


# In[63]:


#reversed string
str1[::-1]


# In[61]:


dir(str)


# In[65]:


#use of split()
print(str1)
str1.split()


# In[85]:


#use of join() method
reviews = ["The product is awesome", "Great Service"]
joined_string = ' '.join(reviews)
joined_string


# In[77]:


str4 = "Hello, How are you?"
' '.join(str4)


# In[87]:


#use of strip() method 
#it removes empty spaces of the string
str5 = "      Hello, How are you?      "
str5


# In[89]:


str5.strip()


# In[97]:


# Example dictionary for product sales analysis
sales_data = {
    "ProductID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "ProductName": ["Laptop", "Mouse", "Keyboard", "Monitor", "Chair", "Desk", "Webcam", "Headphones", "Printer", "Tablet"],
    "Category": ["Electronics", "Accessories", "Accessories", "Electronics", "Furniture", "Furniture", "Electronics", "Accessories", "Electronics", "Electronics"],
    "PriceRange": ["High", "Low", "Low", "Medium", "Medium", "Medium", "Low", "Low", "Medium", "High"],
    "StockAvailable": [15, 100, 75, 20, 10, 8, 50, 60, 25, 12],
}


# In[99]:


for k,v in sales_data.items():
    print(k,set(v), end = ',')
    print('/n')


# In[101]:


# Original reviews dictionary
reviews = {
    "Review1": "The product quality is excellent and delivery was prompt. The product functionality is versatile",
    "Review2": "Good service but the packaging could have been better. The customer service has to improve",
    "Review3": "The product works fine, but the customer support is not very helpful. I rate the product as excellent",
}

# Result dictionary to store analysis of reviews
review_analysis = {}

# Process each review
for key, review in reviews.items():
    # Split the review into words
    words = review.lower().replace('.', '').replace(',', '').split()
    # Create a sub-dictionary with word count and unique words
    review_analysis[key] = {
        "WordCount": len(words),
        "UniqueWords": list(set(words))
    }

review_analysis


# In[107]:


d1 = {"Ram": 170, "Shyam": 160, "Ramya": 150}


# In[111]:


for k in d1.keys():
    print(k)


# In[119]:


for v in d1.values():
    print(v)


# In[121]:


for k,v in d1.items():
    print(k,v)


# In[123]:


d1["john"] = 165
d1


# In[ ]:




