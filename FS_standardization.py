import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/Social_Network_Ads.csv')


## scaling when data sets are normally distributed in other stats terms turning normal distribution to standard normal variate 


# df = df.iloc[:,2:5]


# X_train, X_test, y_train, y_test=train_test_split(df.drop('Purchased', axis=1), df['Purchased'] ,train_size=0.3,random_state=0)


# scaler = StandardScaler()

# ## fit the scaler to the train set it will learn the parameters

# scaler.fit(X_train)

# ## transform train and test sets

# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# ## transforming scaled datas back to dataframe beacause scaled datas are in array format


# X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X_train.columns)
# X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X_test.columns)


# ## now using logistic regression model on unscaled data and scaled data to compare the prediction success rate

# lr = LogisticRegression()
# lr_scaled = LogisticRegression()


# lr.fit(X_train, y_train)
# lr_scaled.fit(X_train_scaled, y_train)


# y_pred = lr.predict(X_test)
# y_pred_scaled = lr_scaled.predict(X_test_scaled)


## testing accuracy score of scaled and nonscaled trained model

# print("accureacy of Non_scaled data:", accuracy_score(y_test, y_pred))
# print("accureacy of scaled data:", accuracy_score(y_test, y_pred_scaled))








