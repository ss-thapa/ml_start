import pandas as pd
import numpy as np
from scipy.stats import shapiro
import scipy.stats as stats 
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/daraz_trasnformed.csv")


## lets have a price_of_1 column from the dataset and perform the hypothesies that the mean price is 600  
## H0 says is mean price is 600 H1 says its more than 600 

pop_mean = 600

pop = df['price_of_1'].dropna()

sample_price = pop.sample(25).values

## check normality using shapiro 

shapiro_price = shapiro(sample_price)

t_statistic, p_value = stats.ttest_1samp(sample_price, pop_mean)

# print("t-statistic:", t_statistic)
# print("p-value:", p_value/2)

# print(shapiro_price)

# sns.kdeplot(sample_price)
# plt.show()




alpha = 0.05

# if p_value < alpha:
#     print("Reject the null hypothesis.")
# else:
#     print("Fail to reject the null hypothesis.")





