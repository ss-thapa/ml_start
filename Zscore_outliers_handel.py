import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/placement1.csv")

# plt.figure(figsize=(16,5))
# plt.subplot(1,2,1)
# sns.distplot(df['cgpa'])

# plt.subplot(1,2,2)
# sns.distplot(df['placement_exam_marks'])

# plt.show()


## here only cgpa is only normally distributed placement_exam_marks is right skewed so will will use zscore method to either remove or cap the outliers


# Finding the boundary values
Highest_allowed = df['cgpa'].mean() + 3*df['cgpa'].std()
Lowest_allowed = df['cgpa'].mean() - 3*df['cgpa'].std()


# Finding the outliers
outlier_df = df[(df['cgpa'] > Highest_allowed) | (df['cgpa'] < Lowest_allowed)]


## trimming method 

new_df = df[(df['cgpa'] < 8.80) & (df['cgpa'] > 5.11)]



# Approach 2

# Calculating the Zscore

df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean())/df['cgpa'].std()


outlier_by_z_df = df[(df['cgpa_zscore'] > 3) | (df['cgpa_zscore'] < -3)]

# Trimming 

new_df_by_z = df[(df['cgpa_zscore'] < 3) & (df['cgpa_zscore'] > -3)]






#### Capping method


upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()

lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()


df['cgpa'] = np.where(
    df['cgpa']>upper_limit,
    upper_limit,
    np.where(
        df['cgpa']<lower_limit,
        lower_limit,
        df['cgpa']
    )
)


print(df['cgpa'].describe())

