# %% prepare data
# import modules
import analysis

# set global variables
sub_ids = ["01", "03", "05", "07", "09", "11", "13", "15", "17", "19", "21", "23", "25", "27", "29",
           "02", "04", "06", "08", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30"]
cond_ids = [1, 2]
block_ids = [1, 2, 4, 6]

# load all data
df = analysis.get_concat_df(sub_ids=sub_ids)
questionair_df = analysis.get_questionair_df()

# filter for specific cond_ids and block_ids
df = df[df["cond_id"].isin(cond_ids) & df["block_id"].isin(block_ids)]

# remove all trials with less than 0.3 and more than 15 seconds response time
#df = analysis.remove_trials(df=df)

# data manipulation and calculation of new parameter
df = analysis.data_manipulation(df=df)

# calculate mean led_distance, mean signed and mean absolute error per sub_id, cond_id, block_id and speaker_distance
means_df = analysis.get_means_df(df=df)

# calculate mean and std of mean led_distance responses
mean_of_means_df = analysis.get_mean_of_means_df(means_df=means_df)

#TODO: response time per trial
#TODO: signed error depending on distance difference to previous stimulus

# %% plot individual results per participant
for sub_id in sub_ids:
    analysis.plot_and_save_data_per_sub(df=df, sub_id=sub_id)

# %% plot all datapoints per cond_id and block_id
analysis.plot_data(df=df, x="speaker_distance", y="led_distance",
                   col="block_id", row="cond_id", hue="sub_id", kind="scatterplot", baseline="one_one")

# %% plot averaged led distance
# plot mean results of each sub_id per cond_id and block_id
analysis.plot_data(df=means_df, x="speaker_distance", y="mean_led_distance",
                   col="block_id", row="cond_id", hue="sub_id", kind="lineplot", baseline="one_one")

# plot mean of mean results with error bars
analysis.plot_with_error_bars(df=mean_of_means_df, x="speaker_distance", y="mean_mean_led_distance",
                              yerr="std_mean_led_distance", col="block_id", row="cond_id", baseline="one_one")

# plot boxplot of mean results
analysis.plot_boxplot(df=means_df, x="speaker_distance", y="mean_led_distance", col="block_id", hue="cond_id", baseline="one_one")

# %% plot averaged estimation errors
# plot mean signed error of each sub_id per cond_id and block_id
analysis.plot_data(df=means_df, x="speaker_distance", y="mean_signed_error",
                   col="block_id", row="cond_id", hue="sub_id", kind="lineplot", baseline="zero")

# plot mean of mean signed error with errorbar
analysis.plot_with_error_bars(df=mean_of_means_df, x="speaker_distance", y="mean_mean_signed_error",
                              yerr="std_mean_signed_error", col="block_id", row="cond_id", baseline="zero")

# plot boxplot of mean signed error
analysis.plot_boxplot(df=means_df, x="speaker_distance", y="mean_signed_error", col="block_id", hue="cond_id", baseline="zero")

# %% show data distribution (histogram, qq-plot, shapiro-wilk test and kolmogrov-smirnoff test)
distribution_df = means_df[(means_df["cond_id"] == 2) & (means_df["block_id"] == 6) & (means_df["speaker_distance"] == 11)]
analysis.show_data_distribution(df=distribution_df, x="mean_led_distance")

# %% fitting mean results of each sub_id
# analysis.plot_data(df=means_df, x="speaker_distance", y="mean_led_distance",
#                    col="block_id", row="cond_id", hue="sub_id", kind="regplot")

# %% predict sample size
analysis.predict_sample_size(group_1=[9.539, 1.483, 15], group_2=[10.615, 0.710, 15], alpha=0.05, power=0.8, alternative="two-sided")

# %% diagnostic plots
diagnostic_df = means_df[(means_df["cond_id"] == 2) & (means_df["block_id"] == 1)]
analysis.create_diagnostic_plots(df=diagnostic_df, x="speaker_distance", y="mean_led_distance")
