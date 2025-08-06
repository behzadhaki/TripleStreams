from scipy.stats import ttest_ind_from_stats

# Mean, standard deviation, and sample size for group 1
mean1 = 59  # example value
std1 = 28    # example value
n1 = 7000     # example value

# Mean, standard deviation, and sample size for group 2
mean2 = 59  # example value
std2 = 27    # example value
n2 = 7000     # example value

# Perform the t-test from statistics
t_stat, p_value = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=n1,
                                       mean2=mean2, std2=std2, nobs2=n2,
                                       equal_var=False)  # Assumes unequal variances

print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.5f}")

# Interpreting the result
if p_value < 0.05:
    print("There is a statistically significant difference between the groups.")
else:
    print("There is no statistically significant difference between the groups.")

