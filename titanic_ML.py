import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions
import scipy.stats as stats 
import statsmodels.api as sm
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


df = sns.load_dataset('titanic')                        

print(df.head())

selected_columns = ['survived', 'age', 'fare']
df1 = df[selected_columns]



## handeling missing values in age 

df1['age'] = df1['age'].fillna(df['age'].mean())

## defining X and y 

X = df1.iloc[:,1:3]
y = df1.iloc[:,0]


X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=0.2,random_state=42)


## plotting age columns 

# sns.distplot(X_train['age'])
# sm.qqplot(X_train['age'], line='45', fit=True)
# plt.show()


## plotting fare column

# sns.distplot(X_train['fare'])
# sm.qqplot(X_train['fare'], line='45', fit=True)
# plt.show()




## traings with 2 models 

# clf = LogisticRegression()
# clf1 = DecisionTreeClassifier()

# clf.fit(X_train, y_train)
# clf1.fit(X_train, y_train)


# y_pred = clf.predict(X_test)
# y_pred1 = clf1.predict(X_test)


# print("Accuracy LR", accuracy_score(y_test, y_pred))
# print("Accuracy LR", accuracy_score(y_test, y_pred1))





## transforming fare column to normal distribution to check the result 

trf = FunctionTransformer(func=np.log1p)

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.fit_transform(X_test)


clf = LogisticRegression()
clf1 = DecisionTreeClassifier()

clf.fit(X_train_transformed, y_train)
clf1.fit(X_train_transformed, y_train)

y_pred = clf.predict(X_test_transformed)
y_pred1 = clf1.predict(X_test_transformed)


# print("Accuracy LR", accuracy_score(y_test, y_pred))
# print("Accuracy DR", accuracy_score(y_test, y_pred1))
