import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
pd.set_option("display.max_columns", None)

df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/customer.csv')


## In this dataset there are three categorical column where gender and purchsed is nominal categorical data and review and education is ordinal categorical data
## and purchsed colum is our output column i.e Y and gender, review and education is our input column i.e X
## we have to use Ordinal encoder for review and education
## we have to use labeled encoder for purchased 
## we have to use one hot encoder for gender 



## using ordinal encoder on input column beaccues they are ordinal categorical data 

# df = df.iloc[:,2:]

# X_train, X_test, y_train, y_test=train_test_split(df.iloc[:,0:2], df.iloc[:,-1] ,test_size=0.2)


# ## now while using ordinal encoder for review and education you can see the order of the category least value will be provided to poor and max value will be
# ## provided for Good there is order for the encoding values also
# ## while passing the list in the categories parameter also keep in the mind the order of the column in the dataframe

# oe = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], ['School', 'UG', 'PG']]) 

# oe.fit(X_train)

# X_train = oe.transform(X_train)
# X_test = oe.transform(X_test)


# ## now using labelencoder for output ordinal categorical data because its the output column

# le = LabelEncoder()

# le.fit(y_train)

# y_train = le.transform(y_train)
# y_test = le.transform(y_test)




### One hot encoding is used for nominal categorical data type 


df1 = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/cars.csv')

## there are 3 categorical columns to be encoded by using onehotencoding first we will encode 2 columns fuel and owner 
## drop_first_column is done because while training ml models the could not build the relationship between these encoded new numerical columns if not done it could create conflict
## or it means to avoide dummy variable trap
## or it we dont do that we introduce multicollinearity among the encoded columns


X_train, X_test, y_train, y_test=train_test_split(df1.iloc[:,0:4], df1.iloc[:,-1] ,test_size=0.2)

ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)      ## sparse_output = False it converts X_train_new and X_test_new tp array

X_train_new = ohe.fit_transform(X_train[['fuel', 'owner']])
X_test_new = ohe.transform(X_test[['fuel', 'owner']])

print(X_train_new)

np.hstack((X_train[['brand', 'km_driven']].values, X_train_new))


## encoding brand column seperatly because there are many categories 

counts = df1['brand'].value_counts()

df1['brand'].nunique()
threshold = 100

replac = counts[counts <= threshold].index

pd.get_dummies(df1['brand'].replace(replac, 'Uncommon'), dtype=np.int32)

