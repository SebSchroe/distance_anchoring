# %% prepare data
# import modules
import analysis
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import LinearRegDiagnostic
import matplotlib.pyplot as plt
import scipy.stats as stats

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

# data treatment and calculation of accuarcy parameter
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
y = "absolute_error"
means_df, mean_of_means_df = analysis.plot_averaged_data(df=cleaned_df, x="speaker_distance", y=y)

# %% plot average accuracy per block
remove_condition = (
    (
        (cleaned_df["block_id"] == 6) &
        (cleaned_df["cond_id"] == 1) &
        (cleaned_df["speaker_distance"].isin([2, 8, 9, 10, 11, 12]))
    ) |
    (
        (cleaned_df["block_id"] == 6) &
        (cleaned_df["cond_id"] == 2) &
        (cleaned_df["speaker_distance"].isin([2, 3, 4, 5, 6, 12]))
    )
)
filtered_df = cleaned_df[~remove_condition] # TODO: its a really cool idea to only look at only a speaker subset in block 6 for "interpolation analysis"
# TODO: maybe also calculate a learning coefficient like the difference value between block 1 and block 6


learning_df = (filtered_df
               .groupby(["sub_id", "cond_id", "block_id"], as_index=False)
               .agg(mean_value=("absolute_error", "mean"))
               )

mean_learning_df = (learning_df
                    .groupby(["cond_id", "block_id"], as_index=False)
                    .agg(mean_mean_value=("mean_value", "mean"),
                         std_mean_value=("mean_value", "std"))
                    )
# for sub_id in learning_df["sub_id"].unique():
#     sub_id_int = int(sub_id)
#     sub_df = learning_df[learning_df["sub_id"] == sub_id_int]
#     analysis.plot_data(df=sub_df, x="block_id", y="mean_value",  col=None, row=None, hue="cond_id", kind="lineplot")

# test hypothesis 2
# set parameters for the model
x = "block_id"
y = "mean_value"
fixed_group = "cond_id"
random_group = "sub_id"

# log-transformation if necessary
learning_df[f"log_{x}"] = np.log(learning_df[x])
learning_df[f"log_{y}"] = np.log(learning_df[y])
x = f"log_{x}"
y = f"log_{y}"

# two way ANOVA with interaction
ANOVA = smf.ols(formula=f"{y} ~ C({x}) * C({fixed_group})", data=learning_df).fit()
print(ANOVA.summary())

# diagnostic plots for LM
cls = LinearRegDiagnostic.LinearRegDiagnostic(ANOVA)
vif, fig, ax = cls()
print(vif)

# mixed effects ANOVA with interaction
ANOVA_mixed_effect = smf.mixedlm(
    formula=f"{y} ~ C({x}) * C({fixed_group})", # fixed effect
    groups=learning_df[fixed_group], # random intercept grouping factor 
    re_formula="~1", # random intercept for each group in random group
    data=learning_df, # data
    ).fit(method=["cg"], reml=True) # fitting the model (use reml=False when AIC is needed)
print(ANOVA_mixed_effect.summary())

AIC = ANOVA_mixed_effect.aic
print(AIC)

resid_var = ANOVA_mixed_effect.scale
group_var = ANOVA_mixed_effect.cov_re.iloc[0, 0]
ICC = group_var / (group_var + resid_var)
print(ICC)

# diagnostic plots for LMM
residuals = ANOVA_mixed_effect.resid
fitted_values = ANOVA_mixed_effect.fittedvalues

# QQ-Plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals")
plt.show()

# Residuals vs fitted
plt.scatter(fitted_values, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# visualise data
sns.lmplot(data=learning_df, x=x, y=y, hue=fixed_group, ci=None, order=1)


analysis.plot_data(df=learning_df, x="block_id", y="mean_value", col="cond_id", row=None, hue="sub_id", kind="lineplot")
analysis.plot_with_error_bars(df=mean_learning_df, x="block_id", y="mean_mean_value", yerr="std_mean_value", col="cond_id", row=None)
analysis.plot_boxplot(df=learning_df, x="cond_id", y="mean_value", col=None, hue="block_id")

# %% t-test at specific speaker distances (Welchs t-test aka Two-sample t_test with different std)
analysis.t_test_speaker_distance(df=df, block_id=6, speaker_distance=3, group_ids=[2, 3], y=y)

# %% general linear model and general linear mixed model for hypothesis 1

# create necessary dataframe
cond_ids = [1, 2, 3]
block_ids = [1]
model_df = means_df[(means_df["cond_id"].isin(cond_ids)) & (means_df["block_id"].isin(block_ids))]

# add the perfect line
# cond_0 = []
# for speaker_distance in model_df["speaker_distance"].unique():
#     cond_0_row = {"sub_id": 0, "cond_id": 0, "block_id": 1, "speaker_distance": speaker_distance, "mean_response_distance": speaker_distance, "std_response_distance": np.nan}
#     cond_0.append(cond_0_row)

# model_df = pd.concat([model_df, pd.DataFrame(cond_0)], ignore_index=True)

# split cond 3 in cond 3 and cond 4
# Entferne die Einträge mit speaker_distance in [2.1, 11.78]
filtered_df = model_df[~model_df["speaker_distance"].isin([2.1, 11.78])].copy()

# Passe die cond_id basierend auf speaker_distance an
filtered_df.loc[filtered_df["speaker_distance"].isin([2.96, 3.84, 4.74, 5.67]), "cond_id"] = 3
filtered_df.loc[filtered_df["speaker_distance"].isin([7.6, 8.8, 9.64, 10.7]), "cond_id"] = 4

# Klone den Eintrag für speaker_distance == 6.62
clone = filtered_df[filtered_df["speaker_distance"] == 6.62].copy()

# Weise einmal cond_id = 3 und einmal cond_id = 4 zu
filtered_df.loc[filtered_df["speaker_distance"] == 6.62, "cond_id"] = 3
clone["cond_id"] = 4

# Füge den geklonten Eintrag wieder hinzu
final_df = pd.concat([filtered_df, clone], ignore_index=True)


# define parameter for the model
x = "speaker_distance"
y= "mean_signed_error"
fixed_group = "cond_id"
random_group = "sub_id"

# sort reference group for modelling
model_df['cond_id'] = model_df['cond_id'].astype('category')
model_df['cond_id'] = model_df['cond_id'].cat.reorder_categories([3, 1, 2], ordered=True)

# log transform data if necessary
model_df[f"log_{x}"] = np.log(model_df[f"{x}"])
model_df[f"log_{y}"] = np.log(model_df[f"{y}"])

x = f"log_{x}"
y= f"log_{y}"

# linear regression model
linear_regression = smf.ols(formula=f"{y} ~ {x}", data=model_df).fit() # formula = y ~ x
print(linear_regression.summary())

# ANCOVA with interaction
ANCOVA = smf.ols(formula=f"{y} ~ {x} * C({fixed_group})", data=model_df).fit()
# print(ANCOVA.summary())

# mixed effect linear regression
linear_mixed_effect = smf.mixedlm(
    formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
    groups=model_df[f"{random_group}"], # random intercept grouping factor 
    re_formula=f"~{x}", # random slope formula
    data=model_df, # data
    ).fit(method=["lbfgs"]) # fitting the model

print(linear_mixed_effect.summary())

resid_var = linear_mixed_effect.scale
group_var = linear_mixed_effect.cov_re.iloc[0, 0]
random_slope_variance = linear_mixed_effect.cov_re.loc[x, x]
ICC = group_var / (group_var + random_slope_variance + resid_var)
print(ICC)

# generate diagnostic plots
cls = LinearRegDiagnostic.LinearRegDiagnostic(ANCOVA)
vif, fig, ax = cls()
print(vif)

# plot data for visualization
sns.lmplot(data=model_df, x=x, y=y, hue=fixed_group, ci=None, order=1) # Takeaway: there is no difference in the fitting of all data or mean data



