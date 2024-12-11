# %% prepare data
# import modules
import analysis
from scipy.stats import shapiro, kstest, ttest_ind

# set global variables
sub_ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
           "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
           "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]
cond_ids = [1, 2]
block_ids = [1, 2, 4, 6]

# load all data
Basti_df = analysis.get_Basti_df(sub_ids=sub_ids, cond_ids=cond_ids, block_ids=block_ids)
Oskar_df = analysis.get_Oskar_df()
df = analysis.merge_dataframes(df_1=Basti_df, df_2=Oskar_df)
questionnaire_df = analysis.get_questionnaire_df()

# data treatment and calculation of error parameter
df = analysis.data_calculations(df=df)

# remove failed trials (less than 0.5 seconds)
df = analysis.remove_failed_responses(df=df)

# identify outliers in response_time with 3 sd method of each block per sub_id
cleaned_df = analysis.identify_and_remove_response_outliers(df=df)

# calculate mean response_distance, mean signed and mean absolute error
means_df = analysis.get_means_df(df=cleaned_df)

# calculate means of mean response_distance, mean signed and mean absolute error
mean_of_means_df = analysis.get_mean_of_means_df(means_df=means_df)

#TODO: signed error depending on distance difference to previous stimulus

#TODO: change speaker_distances in Oskars data so they match my Data

# %% plot individual raw data per participant and save the plot
for sub_id in cleaned_df["sub_id"].unique():
    analysis.plot_data_per_sub(df=cleaned_df, sub_id=sub_id, x="speaker_distance", y="response_distance", baseline="one_one", save=True)

# %% observe questionnaire at its own
analysis.observe_questionnaire(df=questionnaire_df, x="cond_id", y="age", hue="gender")

# %% plot averaged data
analysis.plot_averaged_data(df=cleaned_df, x="speaker_distance", y="response_distance")

# %% t-test (Welchs t-test aka Two-sample t_test with different std)

#TODO: check for property of independence, identically distributed, normally distributed and equal variances

dof = 14
group_1_df = means_df[(means_df["cond_id"] == 1) & (means_df["block_id"] == 6) & (means_df["speaker_distance"] == 12)]
group_1_array = group_1_df["mean_response_distance"]

group_2_df = means_df[(means_df["cond_id"] == 2) & (means_df["block_id"] == 6) & (means_df["speaker_distance"] == 12)]
group_2_array = group_2_df["mean_response_distance"]

t_stat, p_val = ttest_ind(a=group_1_array, b=group_2_array, equal_var=False, alternative="two-sided")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.3f}")

# %% show data distribution (histogram, qq-plot, shapiro-wilk test and kolmogrov-smirnoff test)
distribution_df = means_df[(means_df["cond_id"] == 1) & (means_df["block_id"] == 6) & (means_df["speaker_distance"] == 12)]
analysis.show_data_distribution(df=distribution_df, x="mean_response_distance")

# %% predict sample size
group_1, group_2 = analysis.get_group_parameter(df=mean_of_means_df, block_id=6, speaker_distance=12)
analysis.predict_sample_size(group_1=group_1, group_2=group_2, alpha=0.05, nobs1=15, alternative="two-sided")


# %% diagnostic plots
diagnostic_df = means_df[(means_df["cond_id"] == 1) & (means_df["block_id"] == 6) & (means_df["speaker_distance"] == 12)]
analysis.create_diagnostic_plots(df=diagnostic_df, x="speaker_distance", y="mean_response_distance")
