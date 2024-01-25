import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/covid_toy.csv')


## in this data frame age and fever are numerical column 
## gender and city are nominal categorical columns 
## cough is ordinal categorical column because there is order like mild and strong
## all coulms except has_covid are X i.e input and has_covid is Y i.e output 
## fever has many value missing 


### so from the above desciption we have to do following pre-processing 
### we have to do simpleimputer in fever column
### we have to do OneHotEncoder in gender and city
### we have to do OrdinalEncoder in cough 
### we have to do LabelEncoder for has_covid because its output column
### we also can do scaling in numerical columns 


X_train, X_test, y_train, y_test=train_test_split(df.iloc[:,0:5], df.iloc[:,-1] ,test_size=0.2)

### if we dont use comlumn tranformer we had to do all the simpleimputer,OneHotEncoder,OrdinalEncoder,scaling etc differently and the code would be long 
### we had to do all the tranformation and concatinate 
### but by using ColumnTransformer we can do it quickly and in less code 
## label encoder  cannot be used in the ColumnTransformer



tranformer = ColumnTransformer(transformers=[
    ('tnf1', SimpleImputer(), ['fever']),
    ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
    ('tnf3', OneHotEncoder(sparse_output=False, drop='first'),['gender', 'city'])
], remainder='passthrough')


tranformer.fit_transform(X_train)
tranformer.transform(X_test)


le = LabelEncoder()

le.fit_transform(y_train)
le.transform(y_test)






