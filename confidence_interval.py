import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/daraz_trasnformed.csv")



### using T-procedure

# # Assuming you have already loaded your data into the DataFrame df
# data_column = df['price_of_1']

# num_samples = 100
# sample_size = 35
# sample_means = []

# for _ in range(num_samples):
#     sample = np.random.choice(data_column, size=sample_size, replace=True)
#     sample_mean = np.mean(sample)
#     sample_means.append(sample_mean)

# # Calculate mean and standard deviation of sample means
# sample_mean = np.mean(sample_means)
# sample_std = np.std(sample_means, ddof=1)  # Use ddof=1 for sample standard deviation

# # Choose a confidence level
# confidence_level = 0.95

# # Degrees of freedom for t-distribution
# degrees_of_freedom = sample_size - 1

# # Calculate margin of error using t-distribution values
# t_margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom) * (sample_std / np.sqrt(num_samples))

# # Calculate confidence interval
# lower_limit = sample_mean - t_margin_of_error
# upper_limit = sample_mean + t_margin_of_error

# print(f"Confidence Interval ({confidence_level*100}%): ({lower_limit}, {upper_limit})")







# ##Assuming you have already loaded your data into the DataFrame df
# data_column = df['price_of_1']

# num_samples = 100
# sample_size = 30
# sample_means = []
# sample_stds = []

# for _ in range(num_samples):
#     sample = np.random.choice(data_column, size=sample_size, replace=True)
#     sample_std = np.std(sample)
#     sample_mean = np.mean(sample)
#     sample_stds.append(sample_std)
#     sample_means.append(sample_mean)

# # Calculate mean and standard deviation of sample means
# sample_mean = np.mean(sample_means)
# sample_std = np.mean(sample_stds)

# # Choose a confidence level
# confidence_level = 0.95

# # Degrees of freedom for t-distribution
# degrees_of_freedom = sample_size - 1

# # Calculate margin of error using t-distribution values
# t_margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom) * (sample_std / np.sqrt(num_samples))

# # Calculate confidence interval
# lower_limit = sample_mean - t_margin_of_error
# upper_limit = sample_mean + t_margin_of_error

# print(f"Confidence Interval ({confidence_level*100}%): ({lower_limit}, {upper_limit})")











# data_column = df['price_of_1']

# num_samples = 100
# sample_size = 30
# sample_means = []
# sample_stds = []

# for _ in range(num_samples):
#     sample = np.random.choice(data_column, size=sample_size, replace=True)
#     sample_std = np.std(sample)
#     sample_mean = np.mean(sample)
#     sample_stds.append(sample_std)
#     sample_means.append(sample_mean)

# # Calculate mean and standard deviation of sample means
# sample_mean = np.mean(sample_means)
# sample_std = np.mean(sample_stds)

# # Choose a confidence level
# confidence_level = 0.95


# # Calculate confidence interval
# lower_limit = sample_mean - 2.04523 * (sample_std/np.sqrt(30))
# upper_limit = sample_mean + 2.04523 * (sample_std/np.sqrt(30))

# print(f"Confidence Interval ({confidence_level*100}%): ({lower_limit}, {upper_limit})")


