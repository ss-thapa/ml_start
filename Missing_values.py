import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



# df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/titanic_toy.csv")


# ### Univariate numerical missing value impuatation using mean/median

# X = df.drop(columns=['Survived'])
# y = df['Survived']



# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# imputer1 = SimpleImputer(strategy='median')
# imputer2 = SimpleImputer(strategy='mean')

# trf = ColumnTransformer([
#     ('imputer1',imputer1,['Age']),
#     ('imputer2',imputer2,['Fare'])
# ],remainder='passthrough')

# trf.fit(X_train)

# X_train = trf.transform(X_train)
# X_test = trf.transform(X_test)


# ### univariate numerical missing value imputation using arbitary values

# imputer12 = SimpleImputer(strategy='constant',fill_value=99)
# imputer22 = SimpleImputer(strategy='constant',fill_value=999)

# trf1 = ColumnTransformer([
#     ('imputer12',imputer12,['Age']),
#     ('imputer22',imputer22,['Fare'])
# ],remainder='passthrough')

# trf1.fit(X_train)

# X_train = trf1.transform(X_train)

# X_test = trf1.transform(X_test)

# X_train



### univariate categorical missing value imputation with mode or 


df1 = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/train1.csv",usecols=['GarageQual','FireplaceQu','SalePrice'])

### 47 percent of  FireplaceQu data are missing and 5.5 percent of FireplaceQu data are missing.

X_train,X_test,y_train,y_test = train_test_split(df1.drop(columns=['SalePrice']),df1['SalePrice'],test_size=0.2)


# imputer = SimpleImputer(strategy='most_frequent')

# X_train = imputer.fit_transform(X_train)

# X_test = imputer.transform(X_train)


### univariate categorical missing value imputation with creating missing category

imputer = SimpleImputer(strategy='constant',fill_value='Missing')

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_train)


