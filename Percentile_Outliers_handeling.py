import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/weight-height.csv")


# sns.boxplot(df['Height'])
# plt.show()

## setting the upper limit and lowerlimit with percentile

upper_limit = df['Height'].quantile(0.99)

lower_limit = df['Height'].quantile(0.01)

## trimming the data

new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]



# Capping --> Winsorization

df['Height'] = np.where(df['Height'] >= upper_limit,
        upper_limit,
        np.where(df['Height'] <= lower_limit,
        lower_limit,
        df['Height']))



# sns.boxplot(df['Height'])
# plt.show()