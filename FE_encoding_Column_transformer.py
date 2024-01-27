import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



# df = pd.read_csv('/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/covid_toy.csv')


# ## in this data frame age and fever are numerical column 
# ## gender and city are nominal categorical columns 
# ## cough is ordinal categorical column because there is order like mild and strong
# ## all coulms except has_covid are X i.e input and has_covid is Y i.e output 
# ## fever has many value missing 


# ### so from the above desciption we have to do following pre-processing 
# ### we have to do simpleimputer in fever column
# ### we have to do OneHotEncoder in gender and city
# ### we have to do OrdinalEncoder in cough 
# ### we have to do LabelEncoder for has_covid because its output column
# ### we also can do scaling in numerical columns 


# X_train, X_test, y_train, y_test=train_test_split(df.iloc[:,0:5], df.iloc[:,-1] ,test_size=0.2)

# ### if we dont use comlumn tranformer we had to do all the simpleimputer,OneHotEncoder,OrdinalEncoder,scaling etc differently and the code would be long 
# ### we had to do all the tranformation and concatinate 
# ### but by using ColumnTransformer we can do it quickly and in less code 
# ## label encoder  cannot be used in the ColumnTransformer



# tranformer = ColumnTransformer(transformers=[
#     ('tnf1', SimpleImputer(), ['fever']),
#     ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
#     ('tnf3', OneHotEncoder(sparse_output=False, drop='first'),['gender', 'city'])
# ], remainder='passthrough')


# tranformer.fit_transform(X_train)
# tranformer.transform(X_test)


# le = LabelEncoder()

# le.fit_transform(y_train)
# le.transform(y_test)





### encoding numerical columns by converting those columns into categorical columns 


df1 = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/train.csv", usecols=['Age','Fare','Survived'])


df1.dropna(inplace=True)

X_train, X_test, y_train, y_test=train_test_split(df1.iloc[:,1:], df1.iloc[:,0] ,test_size=0.2, random_state=42)


clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)

np.mean(cross_val_score(DecisionTreeClassifier(),df1.iloc[:,1:],df1.iloc[:,0], cv=10, scoring='accuracy'))


## without applying any sort of tranformation we are getting accuracy score of 62 percent

kbin_age = KBinsDiscretizer(n_bins=15, encode='ordinal',strategy='quantile')
kbin_fare = KBinsDiscretizer(n_bins=15, encode='ordinal',strategy='quantile')

trf = ColumnTransformer([
    ('first',kbin_age,[0]),
    ('second', kbin_fare,[1])
])

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)

trf.named_transformers_['first'].n_bins_     ## to check how much bins are created


clf1 = DecisionTreeClassifier()

clf1.fit(X_train_trf,y_train)
y_pred1 = clf1.predict(X_test_trf)

accuracy_score(y_test,y_pred1)





## creating a user defined function to use diffirent strategy and bins and printing cross val score and plotting before and after graphs of the columns


def discretize(bins,strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy=strategy)
    kbin_fare = KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy=strategy)
    
    trf = ColumnTransformer([
        ('first',kbin_age,[0]),
        ('second',kbin_fare,[1])
    ])
    
    X_trf = trf.fit_transform(df1.iloc[:,1:])
    print(np.mean(cross_val_score(DecisionTreeClassifier(),df1.iloc[:,1:],df1.iloc[:,0],cv=10,scoring='accuracy')))
    
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.hist(df1.iloc[:,1:]['Age'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,0],color='red')
    plt.title("After")

    plt.show()
    
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.hist(df1.iloc[:,1:]['Fare'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,1],color='red')
    plt.title("Fare")

    plt.show()

discretize(15,'uniform')
