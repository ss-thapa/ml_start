import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

pd.set_option("display.max_columns", None)


df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/train.csv')

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


X_train,X_test,Y_train,Y_test=train_test_split(df.iloc[:,1:8], df.iloc[:,0], test_size=0.2, random_state=42)


df.isnull().sum()

### there are missing values in age and embarked and filling missing value of age with mean and of embarked with most frequent because its string


si_age = SimpleImputer()
si_embarked = SimpleImputer(strategy='most_frequent')

X_train_age = si_age.fit_transform(X_train[['Age']])
X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])

X_test_age = si_age.transform(X_test[['Age']])
X_test_embarked = si_embarked.transform(X_test[['Embarked']])



### using onehotencoding on sex and embarked because its nominal categorical data and its in input column
### not performing OneHotEncoder on both column at once because embarked column have missing values
### not dropping first column while encoding because while peforming DecisionTreeClassifier we dont meet dummy variable trap

ohe_sex = OneHotEncoder(sparse_output=False)
ohe_embarked = OneHotEncoder(sparse_output=False)

X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)

X_test_sex = ohe_sex.fit_transform(X_test[['Sex']])
X_test_embarked = ohe_embarked.transform(X_test_embarked)


### concatinate the arrays 


X_train_rem = X_train.drop(columns=['Sex','Age', 'Embarked'])
X_test_rem = X_test.drop(columns=['Sex','Age', 'Embarked'])


X_train_transformed = np.concatenate((X_train_rem,X_train_age,X_train_sex,X_train_embarked), axis=1)
X_test_transformed = np.concatenate((X_test_rem,X_test_age,X_test_sex,X_test_embarked), axis=1)


clf = DecisionTreeClassifier()
clf.fit(X_train_transformed,Y_train)


y_pred = clf.predict(X_test_transformed)


print(accuracy_score(Y_test,y_pred))




