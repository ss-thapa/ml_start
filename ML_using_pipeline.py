import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import set_config
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle


pd.set_option("display.max_columns", None)

df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/train.csv')

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


X_train,X_test,Y_train,Y_test=train_test_split(df.iloc[:,1:8], df.iloc[:,0], test_size=0.2, random_state=42)



## imputation transformer 

trf1 = ColumnTransformer([
    ('impute_age', SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
], remainder='passthrough')

## one hot encoding 2 columns sex and embarked at once 
## i didnt use column name while using OneHotEncoder and SimpleImputer because array dont have column names so always use number indexing

trf2 = ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])
], remainder='passthrough')



## scaling using MinMaxScaler
## till now we have got 10 columns because while doing OneHotEncoder

trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0,10))
])


## feature selection

trf4 = SelectKBest(score_func=chi2, k=8)


## train the model

trf5 = DecisionTreeClassifier()


## creating the pipeline

pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])


### if in the pipeline process doesnt include the model training we have to do fit_transform 
### but in this case we have trf5 which include model train process we use trnasform only


## train the model

pipe.fit(X_train,Y_train)

## display the pipeline

set_config(display='diagram')



## to see the what mean value did SimpleImputer put in
pipe.named_steps['trf1'].transformers_[0][1].statistics_

## to see what most frequent value did SimpleImputer put in 

pipe.named_steps['trf1'].transformers_[1][1].statistics_

## accuracy score of the model

y_pred = pipe.predict(X_test)

accuracy_score(Y_test,y_pred)


### cross validation using pipeline 
### this step what it does is it train test split the mentioned times "cv" and gives the mean reasult of the times

cross_val_score(pipe,X_train,Y_train, cv=5, scoring='accuracy').mean()


## grid search its tuining technique which can either enhance the accuracy or be constatnt and decrease its only used in decision tree


params = {
    'trf5__max_depth':[1,2,3,4,5,None]
}

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, Y_train)

grid.best_score_

grid.best_params_



## Exporting the Pipeline

pickle.dump(pipe, open('pipe.pkl', 'wb'))




### testing my data using the pickle 
### now if we make changes in the above code we have to use new pipe.pkl file we create after change

pipe2 = pickle.load(open('pipe.pkl', 'rb'))

test_input1 = np.array([2, 'male', 31.0,0,0,10.5,'s'], dtype=object).reshape(1,7)

pipe.predict(test_input1)



