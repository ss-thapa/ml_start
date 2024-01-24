import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score





df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/wine_data.csv', header=None, usecols=[0,1,2])
df.columns=['class_label', 'alcohol', 'malic_acid']


# sns.kdeplot(df['alcohol'])
# sm.qqplot(df['alcohol'], line='45', fit=True)
# plt.show()

# sns.kdeplot(df['malic_acid'])
# sm.qqplot(df['malic_acid'], line='45', fit=True)
# plt.show()



## alcohol data is normally distributed for all practical purpose and malic_acid is slightly right skewed looks like log normal distribution


X_train, X_test, y_train, y_test=train_test_split(df.drop('class_label', axis=1), df['class_label'] ,test_size=0.3,random_state=0)



## using min max scaler method to scale the data and transform

scaler = MinMaxScaler()

scaler.fit(X_train)


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X_test.columns)


# print(np.round(X_train).describe(), 1)

# print(np.round(X_test_scaled_df).describe(), 1)

