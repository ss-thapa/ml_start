import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import  scipy.stats as stats 
import statsmodels.api as sm
import psycopg2



df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/placement.csv')

print(df)

# df = df.iloc[:,1:]


# # sns.relplot(kind='scatter', data=df, x='cgpa', y='iq', hue='placement')
# # plt.show()


# ## input and output datas are being seperated
# X = df.iloc[:,0:2]
# Y = df.iloc[:,-1]


# ## train data and testing datas are seperated into variables
# X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.1)


# ## scaling datas
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)


# ## model Training 

# clf = LogisticRegression()

# clf.fit(X_train,Y_train)



# ## predicting and comparing

# y_pred = clf.predict(X_test)




# plot_decision_regions(X_train, Y_train.values, clf=clf, legend=2)
# plt.show()







df = pd.read_csv('/Users/sunilthapa/Desktop/programming/My_final/ML_intro/daraz_trasnformed.csv')

df =df.iloc[:,1:]

## transformed price of 1 column to normal distribution by using log transform

epsilon = 1e-6
df['log_price_of_1'] = np.log(df['price_of_1'] + epsilon)




# sm.qqplot(df['price_of_1'], line='45', fit=True)
# sm.qqplot(df['log_price_of_1'], line='45', fit=True)
# plt.show()

# print(df['price_of_1'].describe())
# print(df['log_price_of_1'].describe())




dbname = "gaming"
user = "postgres"
password = "thapa9860"
host = "localhost"
port = "5432"  # The default port is usually 5432

# Establish a connection to the database
conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Example: Fetch all rows from a table named 'your_table_name'
table_name = "gamingdf"
query = f"SELECT * FROM {table_name};"
cursor.execute(query)

# Fetch the result as a pandas DataFrame
columns = [desc[0] for desc in cursor.description]
data = cursor.fetchall()
df2 = pd.DataFrame(data, columns=columns)

# Close the cursor and connection
cursor.close()
conn.close()




# print(df2['na_sales'].skew())


print(df2.head())


# sns.kdeplot(df2['na_sales'])
# plt.show()

# sm.qqplot(df2['na_sales'], line='45', fit=True)
# plt.show()



