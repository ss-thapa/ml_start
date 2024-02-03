import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import levene
import scipy.stats as stats 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t



### to calculate p value
# Set the t-value and degrees of freedom
t_value = -5.25
df = 58  # Replace this with your specific degrees of freedom

# Calculate the CDF value
cdf_value = t.cdf(t_value, df)
# print(cdf_value*2)        ### *2 is done if the test is 2 tailed if single test then dont do *2 


# df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/daraz_trasnformed.csv")


## lets have a price_of_1 column from the dataset and perform the hypothesies that the mean price is 600  
## H0 says is mean price is 600 H1 says its more than 600 

# pop_mean = 600

# pop = df['price_of_1'].dropna()

# sample_price = pop.sample(25).values

# ## check normality using shapiro 

# shapiro_price = shapiro(sample_price)

# t_statistic, p_value = stats.ttest_1samp(sample_price, pop_mean)

# # print("t-statistic:", t_statistic)
# # print("p-value:", p_value/2)

# # print(shapiro_price)

# # sns.kdeplot(sample_price)
# # plt.show()




# alpha = 0.05

# # if p_value < alpha:
# #     print("Reject the null hypothesis.")
# # else:
# #     print("Fail to reject the null hypothesis.")





### independent 2 sample test 


# df['brand_category'] = df['brand_name'].apply(lambda x: 'branded' if x != 'Other' else 'non-branded')


# ## now i want to perform a hypothesis that the average price of branded clothes are higher than non-branded clothes
# ## the average price of branded clothes is higher than average price of non-branded colothes
# ### H0 says average price of branded colthes and non-branded clothes are equal 
# ### H1 says average price of branded is higher than non-branded clothes 


# pop_branded = df[df['brand_category'] == 'branded']['price_of_1']
# pop_non_branded = df[df['brand_category'] == 'non-branded']['price_of_1']


# sample_branded = pop_branded.sample(25)
# sample_non_branded = pop_non_branded.sample(25)



# ### checking the distribution of the both samples

# shapiro_branded = shapiro(sample_branded)
# shapiro_non_branded = shapiro(sample_non_branded)


# ### checking the variacne of both columns 

# levene_test = levene(sample_branded,sample_non_branded)


# ### calculate the values of T using 

# t_statistics, p_value = stats.ttest_ind(sample_branded,sample_non_branded)

# print("t_statistics", t_statistics)
# print("p_value", p_value/2)


# alpha = 0.05

# if p_value < alpha:
#     print("Reject the null hypothesis.")
# else:
#     print("Fail to reject the null hypothesis.")




### paired 2 sample t - tests 
### before and after is the weight of the persons before training and after training 


before = np.array([80, 92, 75, 68, 85, 78, 73, 90, 70, 88, 76, 84, 82, 77, 91])
after = np.array([78, 93, 81, 67, 88, 76, 74, 91, 69, 88, 77, 81, 80, 79, 88])



differences = after - before


# plt.hist(differences)
# plt.title("Histogram of Weight Differences")
# plt.xlabel("Weight Differences (kg)")
# plt.ylabel("Frequency")
# plt.show()

shapiro_test = stats.shapiro(differences)
# print("Shapiro-Wilk test:", shapiro_test)

mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)



t_statistics, p_value = stats.ttest_rel(before,after)

# print("T-statistic:", t_statistics)
# print("P-value:", p_value)



# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between 'before' and 'after'.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between 'before' and 'after'.")




