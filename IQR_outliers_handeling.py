import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/placement1.csv")


## here placement_exam_marks is skewed so we will use IQR method to remove or cap the outliers 

# Finding the IQR

percentile25 = df['placement_exam_marks'].quantile(0.25)

percentile75 = df['placement_exam_marks'].quantile(0.75)


iqr = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

## finding outliers 

outliers_df = df[df['placement_exam_marks'] > upper_limit]


### Trimming method

new_df = df[df['placement_exam_marks'] < upper_limit]



### capping method 
### here what we are doing is is the values are greater then upper limit replace those values with upper limit and if the values are lower then lower limit then replace those with lowerlimit if both condition are not true then keep as it is


new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)



# Comparing

# plt.figure(figsize=(16,8))
# plt.subplot(2,2,1)
# sns.distplot(df['placement_exam_marks'])

# plt.subplot(2,2,2)
# sns.boxplot(df['placement_exam_marks'])

# plt.subplot(2,2,3)
# sns.distplot(new_df_cap['placement_exam_marks'])

# plt.subplot(2,2,4)
# sns.boxplot(new_df_cap['placement_exam_marks'])

# plt.show()




