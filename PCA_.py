from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np







### performing PCA using Scikit learn reduced the dimensions from 3D to 2D 


# # Sample dataset with numerical and categorical columns
# data = pd.DataFrame({
#     'numerical_feature1': [1, 2, 3, 4, 5],
#     'numerical_feature2': [10, 20, 30, 40, 50],
#     'categorical_feature': ['A', 'B', 'A', 'C', 'B']
# })

# # Define preprocessing steps for numerical and categorical columns
# numeric_features = ['numerical_feature1', 'numerical_feature2']
# categorical_features = ['categorical_feature']

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder())
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Define PCA
# pca = PCA(n_components=2)

# # Combine preprocessing and PCA into a single pipeline
# pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('pca', pca)])

# # Fit pipeline to data
# pipeline.fit(data)

# # Transform data using the pipeline
# transformed_data = pipeline.transform(data)


# print(transformed_data)
# print("Transformed Data Shape:", transformed_data.shape)




#### Performing PCA in MNIST dataset using dataset



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/train_MNIST.csv")


### showing a row what image is that

# plt.imshow(df.iloc[27203,1:].values.reshape(28,28))
# plt.show()


### model trainning without performing PCA


X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# knn = KNeighborsClassifier()


# knn.fit(X_train,y_train)

# y_pred = knn.predict(X_test)


# print(accuracy_score(y_test,y_pred,))




### now training our model using PCA and checking the accuracy score 


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



pca = PCA(n_components=None)




X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)


accuracy_score(y_test,y_pred,)



### finding optimum num of PCA for this data set and plotting to the graph


# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.show()



 