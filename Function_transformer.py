import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import scipy.stats as stats   
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer 




# df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/CSVs/train.csv', usecols=['Age', 'Fare', 'Survived'])


# df['Age'].fillna(df['Age'].mean(), inplace=True)


# X_train, X_test, y_train, y_test=train_test_split(df.iloc[:,1:3], df.iloc[:,0],test_size=0.3,random_state=0)


# sns.kdeplot(X_train['Age'])
# plt.show()
# stats.probplot(X_train['Age'], dist='norm', plot=plt)
# plt.show()


# sns.kdeplot(X_train['Fare'])
# plt.show()
# stats.probplot(X_train['Fare'], dist='norm', plot=plt)
# plt.show()



# trf1 = ColumnTransformer([
#     ('impute_age', FunctionTransformer(func=np.log1p),[0,1])
# ], remainder='passthrough')


df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/CSVs/concrete_data.csv')

print(df.head())

# X = df.drop('Strength', axis=1)
# Y = df.iloc[:,-1]

# X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)


# ## Applying regression without any transformation

# lr = LinearRegression()

# lr.fit(X_train, Y_train)

# y_pred = lr.predict(X_test)

# r2_score(Y_test, y_pred)



# #### applying box-cox transformation 

# pt = PowerTransformer(method='box-cox')

# ## adding 0.000001 because box-cox doesn't support zero and negative value
# ## here we are converting all 8 columns using box-cox 

# X_train_transformed = pt.fit_transform(X_train + 0.0000001)
# X_test_tranformed = pt.transform(X_test + 0.0000001)


# ## applying regression with tranformation

# lr = LinearRegression()

# lr.fit(X_train_transformed, Y_train)

# y_pred = lr.predict(X_test_tranformed)

# r2_score(Y_test, y_pred)




# ### applying Yeo-johnson transform

# py = PowerTransformer(method='yeo-johnson')

# X_train_transformed_y = py.fit_transform(X_train)
# X_test_tranformed_y = py.transform(X_test)





# ## applying regression with tranformation with yeo-johnson

# lr = LinearRegression()

# lr.fit(X_train_transformed_y, Y_train)

# y_pred_y = lr.predict(X_test_tranformed_y)

# r2_score(Y_test, y_pred_y)


