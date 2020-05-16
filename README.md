# House-Price-Prediction
Import the turicreate tool
import turicreate
then import the data
sales=turicreate.SFrame('home_data.sframe')
we can take a look to the given data 
sales.show()
we can see few lines by 
sales.head()
can also look at the end of the data by
sales.tail()
then make a scatter graph keeping on x axis sqft_living and at y the price
x=sales['sqft_living']
y=sales['price']
turicreate.visualization.scatter(x,y,)
then did the random splitting in training data and the test data, by the training data our model  can about the error that is occouring and can improve itself , we do rndom splitting by
train_data,test_data=sales.random_split(.8,seed=0)
we will do the linera regression for just one ferature sqft_living
sqft_model=turicreate.linear_regression.create(train_data,target='price',features=['sqft_living'])
we can find the average by
test_data['price'].mean()
To see how our prediction look we will import matplotlib and name it as plt
import matplotlib.pyplot as plt
%matplotlib inline
Then we will make a curve for our regression model
plt.plot(test_data['sqft_living'],test_data['price'],'.',test_data['sqft_living'],sqft_model.predict(test_data),'-')

we can increase our feature to one to more
my_features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']

can see the categorical view
sales[my_features].show()
again will apply linear regression
my_features_model=turicreate.linear_regression.create(train_data,target='price',features=my_features)
then apply boosted tree regression , decision tree regression  and random forest and find out their respected rmse, then we pick  that regression which have lowest rmse , here i.e linera regression.
We can find out the most expensive place, by taking that particular zipcode we can find the avg rate of that place.
Further we can add some additional features and then find the rmse which in my sample is lesser.because the house price depend upon more than one feature.



