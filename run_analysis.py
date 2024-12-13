# %% prepare data
# import modules
import analysis
from scipy.stats import shapiro, kstest, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

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

#TODO: signed error depending on distance difference to previous stimulus

# %% plot individual raw data per participant and save the plot
for sub_id in cleaned_df["sub_id"].unique():
    analysis.plot_data_per_sub(df=cleaned_df, sub_id=sub_id, x="speaker_distance", y="response_distance", baseline="one_one", save=True)

# %% observe questionnaire at its own
analysis.observe_questionnaire(df=questionnaire_df, x="cond_id", y="age", hue="gender")

# %% plot averaged data
y = "response_distance"
means_df, mean_of_means_df = analysis.plot_averaged_data(df=cleaned_df, x="speaker_distance", y=y)

# %% t-test (Welchs t-test aka Two-sample t_test with different std)

# TODO: Make a function out of this and make it more interactive -> check every assumption and ask for outlier removal if necessary

# set variables
block_id = 6
speaker_distance = 11
group_cond_ids = [1, 2]

# get data
t_test_df = means_df[(means_df["cond_id"].isin(group_cond_ids)) & (means_df["block_id"] == block_id) & (means_df["speaker_distance"] == speaker_distance)]

# Assumptions:
print("\nAssumption 1 (Independence): Each subject only belong to one group. -> True")
print("Assumption 2 (Outliers): The data of each group have no significant outliers.")
print("Assumption 3 (Normality): The data of each group should be normal distributed.")

sns.boxplot(t_test_df, x="cond_id", y=f"mean_{y}")
sns.swarmplot(t_test_df, x="cond_id", y=f"mean_{y}")
plt.show()

    # 3. Normality: The data for each group should be approximately normal distributed
# check assumptions 2 and 3 for group 1
group_1_df = analysis.detect_and_remove_outliers_with_IQR(df=t_test_df, cond_id=group_cond_ids[0], y=y)
analysis.show_data_distribution(df=group_1_df, x=f"mean_{y}")

# check assumptions 2 and 3 for group 2
group_2_df = analysis.detect_and_remove_outliers_with_IQR(df=t_test_df, cond_id=group_cond_ids[1], y=y)
analysis.show_data_distribution(df=group_2_df, x=f"mean_{y}")

# get array per group
group_1_array = t_test_df[t_test_df["cond_id"] == group_cond_ids[0]][f"mean_{y}"].to_numpy()
group_2_array = t_test_df[t_test_df["cond_id"] == group_cond_ids[1]][f"mean_{y}"].to_numpy()

# Welchs t_test
t_stat, p_val = ttest_ind(a=group_1_array, b=group_2_array, equal_var=False, alternative="two-sided")
print(f"\nT-Test results for conditions {group_cond_ids} in block {block_id} at speaker distance {speaker_distance}.")
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
