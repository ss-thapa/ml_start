import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/daraz_trasnformed.csv")




data_column = df['price_of_1']


num_samples = 100

sample_size = 35


sample_means = []

# Generate samples and calculate means
for _ in range(num_samples):
    sample = np.random.choice(data_column, size=sample_size, replace=True)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)


### converted to normal theoram 

# sns.kdeplot(sample_means)
# plt.show()



# Calculate mean and standard deviation of sample means
sample_mean = np.mean(sample_means)
sample_std = np.std(sample_means, ddof=1)  # Use ddof=1 for sample standard deviation

# Choose a confidence level
confidence_level = 0.95

# Find the critical value (two-tailed)
margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * np.sqrt(len(sample_means))

# Calculate confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"Confidence Interval ({confidence_level*100}%): {confidence_interval}")


### Confidence Interval (95.0%): (767.1566541159784, 807.1282030268787) this result means 95 percent of the time the mean of the population will lie in between (767.1566541159784, 807.1282030268787)
