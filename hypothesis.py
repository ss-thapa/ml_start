import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import levene
import scipy.stats as stats 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols 
 


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


# before = np.array([80, 92, 75, 68, 85, 78, 73, 90, 70, 88, 76, 84, 82, 77, 91])
# after = np.array([78, 93, 81, 67, 88, 76, 74, 91, 69, 88, 77, 81, 80, 79, 88])



# differences = after - before


# # plt.hist(differences)
# # plt.title("Histogram of Weight Differences")
# # plt.xlabel("Weight Differences (kg)")
# # plt.ylabel("Frequency")
# # plt.show()

# shapiro_test = stats.shapiro(differences)
# # print("Shapiro-Wilk test:", shapiro_test)

# mean_diff = np.mean(differences)
# std_diff = np.std(differences, ddof=1)



# t_statistics, p_value = stats.ttest_rel(before,after)

# # print("T-statistic:", t_statistics)
# # print("P-value:", p_value)



# # Interpretation
# alpha = 0.05
# if p_value < alpha:
#     print("Reject the null hypothesis. There is a significant difference between 'before' and 'after'.")
# else:
#     print("Fail to reject the null hypothesis. There is no significant difference between 'before' and 'after'.")




### chi square Chi-Square Test for Independence



# np.random.seed(42)

# ## Data Generation: Simulating data for the example
# gender = np.random.choice(['Male', 'Female'], size=200)
# genre = np.random.choice(['Action', 'Comedy', 'Drama'], size=200)

# ## Create a pandas DataFrame
# data = {'Gender': gender, 'Genre': genre}
# df = pd.DataFrame(data)

# ## Contingency Table
# contingency_table = pd.crosstab(df['Gender'], df['Genre'])

# ## Hypotheses
# ## Null Hypothesis (H0): There is no association between gender and favorite movie genre.
# ## Alternative Hypothesis (H1): There is an association between gender and favorite movie genre.

# ## Perform the Chi-Square Test for Independence
# ## degree of freedom = (Number of Rows−1)×(Number of Columns−1)
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# # print(contingency_table)
# # print(expected)
# # print(dof)

# ##Display results
# print("Chi-Square Test Statistic:", chi2)
# print("P-value:", p)
# print("Degrees of Freedom:", dof)
# print("Expected Frequencies Table:")
# print(pd.DataFrame(expected, index=['Male', 'Female'], columns=['Action', 'Comedy', 'Drama']))

# ## Decision
# alpha = 0.05
# print("\nSignificance Level (alpha):", alpha)
# print("P-value < alpha? ", p < alpha)

# if p < alpha:
#     print("Reject the null hypothesis. There is a significant association between gender and favorite movie genre.")
# else:
#     print("Fail to reject the null hypothesis. There is no significant association between gender and favorite movie genre.")







### chi square Goodness-of-Fit Test



# # Observed sales counts
# observed_counts = np.array([70, 50, 80])

# # Expected counts assuming equal contribution
# expected_counts = np.array([66.67, 66.67, 66.67])

# # Perform the Chi-Square Goodness-of-Fit Test
# chi2, p, dof, expected = chi2_contingency([observed_counts, expected_counts])

# # Display results
# print("Chi-Square Test Statistic:", chi2)
# print("P-value:", p)

# # Decision
# alpha = 0.05
# print("\nSignificance Level (alpha):", alpha)
# print("P-value < alpha? ", p < alpha)

# if p < alpha:
#     print("Reject the null hypothesis. The observed sales distribution does not match the expected distribution.")
# else:
#     print("Fail to reject the null hypothesis. The observed sales distribution matches the expected distribution.")






### anova test : one way anova

## Null Hypothesis (H0) :The mean reduction of blood pressure is the same across all three drug groups
## Alternative Hypothesis (H1) : At least one drug group has a different mean reduction of blood pressure compared to the others.



# ##Blood pressure reduction data for each drug group
# drug_a = [10, 12, 15, 11, 14]
# drug_b = [8, 9, 10, 7, 11]
# drug_c = [5, 6, 7, 8, 4]

# ## Perform one-way ANOVA
# statistic, p_value = stats.f_oneway(drug_a, drug_b, drug_c)

# ## Display results
# print("One-Way ANOVA Statistic:", statistic)
# print("P-value:", p_value)

# ## Decision
# alpha = 0.05
# print("\nSignificance Level (alpha):", alpha)
# print("P-value < alpha? ", p_value < alpha)

# if p_value < alpha:
#     print("Reject the null hypothesis. There are significant differences in blood pressure reduction among the drug groups.")
# else:
#     print("Fail to reject the null hypothesis. There are no significant differences in blood pressure reduction among the drug groups.")




### repeated measure anova


# intervention_a = [[5, 6, 4, 7, 5], [4, 5, 6, 3, 4], [6, 7, 5, 6, 7]]
# intervention_b = [[6, 7, 8, 5, 6], [7, 6, 5, 8, 7], [8, 9, 7, 8, 9]]
# intervention_c = [[4, 5, 3, 6, 4], [3, 4, 5, 2, 3], [5, 6, 4, 5, 6]]

# ## Flatten the data (required format for repeated measures ANOVA)
# data = [np.array(intervention_a).flatten(), np.array(intervention_b).flatten(), np.array(intervention_c).flatten()]

# # Ensure all arrays have the same length by padding with NaN values
# max_length = max(len(intervention_a[0]), len(intervention_b[0]), len(intervention_c[0]))
# padded_a = [session + [np.nan] * (max_length - len(session)) for session in intervention_a]
# padded_b = [session + [np.nan] * (max_length - len(session)) for session in intervention_b]
# padded_c = [session + [np.nan] * (max_length - len(session)) for session in intervention_c]

# # Combine all data into a single list
# all_data = padded_a + padded_b + padded_c

# # Create a DataFrame
# df = pd.DataFrame(all_data, columns=[f'Session {i+1}' for i in range(max_length)])
# df['Intervention'] = ['Intervention A'] * 3 + ['Intervention B'] * 3 + ['Intervention C'] * 3

# ## Perform repeated measures ANOVA
# statistic, p_value = stats.f_oneway(*data)

# ## Display results
# print("Repeated Measures ANOVA Statistic:", statistic)
# print("P-value:", p_value)

# ## Decision
# alpha = 0.05
# print("\nSignificance Level (alpha):", alpha)
# print("P-value < alpha? ", p_value < alpha)

# if p_value < alpha:
#     print("Reject the null hypothesis. There are significant differences in anxiety reduction among the therapeutic interventions.")
# else:
#     print("Fail to reject the null hypothesis. There are no significant differences in anxiety reduction among the therapeutic interventions.")




### factorial anova


# Weight loss data for each combination of exercise and diet
no_exercise_standard = [3, 4, 3, 5, 4]
no_exercise_low_carb = [4, 5, 4, 6, 5]
moderate_exercise_standard = [5, 6, 5, 7, 6]
moderate_exercise_low_carb = [6, 7, 6, 8, 7]
intense_exercise_standard = [7, 8, 7, 9, 8]
intense_exercise_low_carb = [8, 9, 8, 10, 9]

# Combine data into a DataFrame
data = {
    'WeightLoss': no_exercise_standard + no_exercise_low_carb + moderate_exercise_standard + moderate_exercise_low_carb + intense_exercise_standard + intense_exercise_low_carb,
    'Exercise': ['No Exercise'] * 5 + ['No Exercise'] * 5 + ['Moderate Exercise'] * 5 + ['Moderate Exercise'] * 5 + ['Intense Exercise'] * 5 + ['Intense Exercise'] * 5,
    'Diet': ['Standard Diet'] * 5 + ['Low Carb Diet'] * 5 + ['Standard Diet'] * 5 + ['Low Carb Diet'] * 5 + ['Standard Diet'] * 5 + ['Low Carb Diet'] * 5
}
df = pd.DataFrame(data)


# Perform factorial ANOVA
formula = 'WeightLoss ~ C(Exercise) * C(Diet)'
model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Display ANOVA results
print(anova_table)

# Decision
alpha = 0.05
interaction_p_value = anova_table.loc['C(Exercise):C(Diet)', 'PR(>F)']
print("\nInteraction Effect p-value:", interaction_p_value)

if interaction_p_value < alpha:
    print("Reject the null hypothesis. There is a significant interaction effect between exercise and diet on weight loss.")
else:
    print("Fail to reject the null hypothesis. There is no significant interaction effect between exercise and diet on weight loss.")
