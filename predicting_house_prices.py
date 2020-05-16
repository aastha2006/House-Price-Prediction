#!/usr/bin/env python
# coding: utf-8

# In[52]:


import turicreate


# In[53]:


sales=turicreate.SFrame('home_data.sframe')


# In[54]:


sales


# In[55]:


x=sales['sqft_living']
y=sales['price']
turicreate.visualization.scatter(x,y,)


# In[6]:


sales.show(x, y,
                xlabel="sqft_living",
                ylabel="Custom Y label",)


# In[7]:


turicreare.show( xlabel="sqft_living", ylabel="price")


# In[8]:


sales['x']=sales['sqft_living']
sales['y']=sales['price']


# In[ ]:


ac=turicreate.visualization.scatter(sales['x'], sales['y'],)


# In[4]:


train_data,test_data=sales.random_split(.8,seed=0)


# In[5]:


sqft_model=turicreate.linear_regression.create(train_data,target='price',features=['sqft_living'])


#  # Evaluate the simle model

# In[6]:


test_data['price'].mean()


# # let's show your predictions look like
# 

# In[7]:


import matplotlib.pyplot as plt


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


plt.plot(test_data['sqft_living'],test_data['price'],'.',test_data['sqft_living'],sqft_model.predict(test_data),'-')


# In[10]:


sqft_model.coefficients


# In[11]:


my_features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']


# In[12]:


sales[my_features].show()


# In[13]:


x=sales['zipcode']
y=sales['price']


# In[14]:


my_features_model=turicreate.linear_regression.create(train_data,target='price',features=my_features)


# In[15]:


x=sales[sales[['zipcode']=='98039']
        


# In[16]:


print(sqft_model.evaluate(test_data))
print(my_features_model.evaluate(test_data))


# 
# <img src="house2.jpg">

# In[17]:


house2 = sales[sales['id']=='1925069082']
house2


# In[51]:





# In[50]:





# <img src="bill_gates.png">

# In[18]:


zipcodes = sales[sales['zipcode']=='98039']


# In[19]:


zipcodes['price'].mean()


# In[20]:


x['price'].mean()


# In[102]:


sales.filter_by((range(2000,4000), 'sqft_living')


# In[21]:


houses= sales.filter_by(filter(lambda x:2000<x<4000, sales['sqft_living']),'sqft_living')
              


# In[109]:


houses


# In[22]:


len(houses)*1.0/len(sales)


# In[23]:


sales


# In[24]:


9111/21613


# In[25]:


advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[26]:


advanced_features_model=turicreate.linear_regression.create(train_data,target='price',features=advanced_features)


# In[27]:


advanced_features_model.evaluate(test_data)


# In[118]:


155356- 179542


# In[28]:



model=turicreate.decision_tree_regression.create(train_data,target='price',features=my_features)       


# In[29]:


model.evaluate(test_data)


# In[41]:


rmodel=turicreate.boosted_trees_regression.create(train_data,target='price',features=my_features)  


# In[48]:


rmodel=turicreate.boosted_trees_regression.create(train_data,target='price',features=my_features)  


# In[43]:


rfmodel=turicreate.random_forest_regression.create(train_data,target='price',features=my_features)  


# In[49]:


model.evaluate(test_data)
rmodel.evaluate(test_data)
rfmodel.evaluate(test_data)


# In[50]:


rmodel.evaluate(test_data)


# In[51]:


rfmodel.evaluate(test_data)


# In[ ]:




