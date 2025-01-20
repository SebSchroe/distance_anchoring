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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.tsaplots import plot_acf

# for Python R Bridge
import os
os.environ["R_HOME"] = "C:/Program Files/R/R-4.4.2" # set R directory

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# import R packages
ro.r('''
library(lme4)
library(emmeans)
     ''')

# activate pandas to R dataframe
pandas2ri.activate()

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

# %% plot averaged data
y = "response_distance"
means_df, mean_of_means_df = analysis.plot_averaged_data(df=cleaned_df, x="speaker_distance", y=y)

# %% plot individual raw data per participant and save the plot
for sub_id in cleaned_df["sub_id"].unique():
    analysis.plot_data_per_sub(df=cleaned_df, sub_id=sub_id, x="speaker_distance", y="response_distance", baseline="one_one", save=True)

# %% observe questionnaire at its own
analysis.observe_questionnaire(df=questionnaire_df, x="cond_id", y="age", hue="gender")

# %% hypothesis 1 analysis
# get dataframe as needed
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean="signed_error")
model_df = means_df[(means_df["cond_id"].isin([1, 2, 3])) & (means_df["block_id"].isin([1]))]

# add the perfect line
# cond_0 = []
# for speaker_distance in model_df["speaker_distance"].unique():
#     cond_0_row = {"sub_id": 0, "cond_id": 0, "block_id": 1, "speaker_distance": speaker_distance, "mean_response_distance": speaker_distance, "std_response_distance": np.nan}
#     cond_0.append(cond_0_row)

# model_df = pd.concat([model_df, pd.DataFrame(cond_0)], ignore_index=True)

# define parameter for the model
x = "speaker_distance"
y= "mean_signed_error"
fixed_group = "cond_id"
random_group = "sub_id"

# sort reference group for modelling
model_df['cond_id'] = model_df['cond_id'].astype('category')
model_df['cond_id'] = model_df['cond_id'].cat.reorder_categories([3, 1, 2], ordered=True)

# log transform data if necessary
# model_df[f"log_{x}"] = np.log(model_df[f"{x}"])
# model_df[f"log_{y}"] = np.log(model_df[f"{y}"])

# x = f"log_{x}"
# y= f"log_{y}"

# ANCOVA with interaction
ANCOVA = smf.ols(formula=f"{y} ~ {x} * C({fixed_group})", data=model_df).fit()
print(ANCOVA.summary())

# generate diagnostic plots
cls = LinearRegDiagnostic.LinearRegDiagnostic(ANCOVA)
vif, fig, ax = cls()
print(vif)

# R model
# model = lmer(y ~ x * fixed_group categorical + (x | random_group))

# mixed effect ANCOVA with interaction and random slope, intercept
model_1 = smf.mixedlm(
    formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
    groups=model_df[f"{random_group}"], # random intercept grouping factor 
    re_formula=f"~{x}", # random slope formula
    data=model_df, # data
    ).fit(method=["lbfgs"]) # fitting the model

print(model_1.summary())

# diagnostic plots for LMM
residuals = model_1.resid
fitted_values = model_1.fittedvalues

# linearity of the predictor
sns.residplot(x=model_df[x], y=residuals, lowess=True, line_kws=dict(color="r"))
plt.xlabel("speaker_distance")
plt.ylabel("Residuals")
plt.title("Residuals vs speaker_distance (linearity)")
plt.show()

# QQ-Plot - normality
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals (normality)")
plt.show()

# Residuals vs fitted - homoscedasticity
sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws=dict(color="r"))
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (homoscedasticity)")
plt.show()

# independency of residuals
plot_acf(residuals)
plt.title("Autocorrelation of Residuals (independency)")
plt.show()

# calculate ICC
resid_var = model_1.scale
group_var = model_1.cov_re.iloc[0, 0]
random_slope_variance = model_1.cov_re.loc[x, x]
ICC = group_var / (group_var + random_slope_variance + resid_var)
print(ICC)

# plot data for visualization
sns.lmplot(data=model_df, x=x, y=y, hue=fixed_group, ci=None, order=1) # Takeaway: there is no difference in the fitting of all data or mean data


# %% hypothesis 2 analysis
# get dataframe as it is needed (for block 6 only speaker of speaker subset used)
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
filtered_df = cleaned_df[~remove_condition]

learning_df = (filtered_df
               .groupby(["sub_id", "cond_id", "block_id"], as_index=False)
               .agg(mean_value=("absolute_error", "mean"))
               )
learning_df = learning_df.rename(columns={"mean_value": "mean_absolute_error"})

# test hypothesis 2
# set parameters for the model
x = "block_id"
y = "mean_absolute_error"
fixed_group = "cond_id"
random_group = "sub_id"

# log-transformation if necessary
learning_df[f"log_{y}"] = np.log(learning_df[y])
y = f"log_{y}"

# set parameters as categorical
# learning_df["block_id"] = learning_df["block_id"].astype("category")
# learning_df["cond_id"] = learning_df["cond_id"].astype("category")
# learning_df["sub_id"] = learning_df["sub_id"].astype("category")

# transfer pandas df to R
ro.globalenv["learning_df"] = pandas2ri.py2rpy(learning_df)
print(ro.r("head(learning_df)")) # check dataframe

# mixed effect two way ANOVA with ineraction in R
ro.r('''
# set independent variables as factor
learning_df$block_id <- as.factor(learning_df$block_id)
learning_df$cond_id <- as.factor(learning_df$cond_id)

# craft the model     
model <- lmer(log_mean_absolute_error ~ block_id * cond_id + (1|sub_id), data = learning_df)

# tukey hsd post hoc test
tukey <- emmeans(model, pairwise ~ block_id * cond_id, adjust = "tukey")
''')

# extrakt results tables from R
print("\nMixed effect ANOVA table:")
print(ro.r("anova(model)"))
print("\nMixed effect ANOVA summary table:")
print(ro.r("summary(model)"))
print("\nTukey HSD post-hoc-test table:")
print(ro.r("summary(tukey)"))


# mixed effects ANOVA with interaction
mixed_effect_ANOVA = smf.mixedlm(
    formula=f"{y} ~ C({x}) * C({fixed_group})", # fixed effect
    groups=learning_df[fixed_group], # random intercept grouping factor 
    re_formula="~1", # random intercept for each group in random group
    data=learning_df, # data
    ).fit(method=["lbfgs"], reml=True) # fitting the model (use reml=False when AIC is needed)
print(mixed_effect_ANOVA.summary())

resid_var = mixed_effect_ANOVA.scale
group_var = mixed_effect_ANOVA.cov_re.iloc[0, 0]
ICC = group_var / (group_var + resid_var)
print(ICC)

# diagnostic plots for LMM
residuals = mixed_effect_ANOVA.resid
fitted_values = mixed_effect_ANOVA.fittedvalues

# QQ-Plot - normality
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals (normality)")
plt.show()

# Residuals vs fitted - homoscedasticity
sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws=dict(color="r"))
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (homoscedasticity)")
plt.show()

# independency of residuals
plot_acf(residuals)
plt.title("Autocorrelation of Residuals (independency)")
plt.show()

# visualise data
analysis.plot_boxplot(df=learning_df, x="cond_id", y=y, col=None, hue="block_id")

# %% hypothesis 3 analysis

# TODO: thoughts: I do two ANCOVA: 1. compare cond_1 block 6 -> 7 - 11 with cond_2 block 6 -> 7 - 11 and cond_2 block 1 vice versa
# second half
include_condition = (
    (
         (cleaned_df["cond_id"].isin([1, 2])) &
         (cleaned_df["block_id"] == 6) &
         (cleaned_df["speaker_distance"].isin([7, 8, 9, 10, 11]))
     ) |
    
    (
        (cleaned_df["cond_id"] == 2) &
        (cleaned_df["block_id"] == 1)
    )
)
filtered_df = cleaned_df[include_condition]

model_3_df = analysis.get_means_df(filtered_df, "response_distance")

# change condition names
model_3_df["condition"] = None
model_3_df.loc[(model_3_df["block_id"] == 1) & (model_3_df["cond_id"] == 2), "condition"] = "naive"
model_3_df.loc[(model_3_df["block_id"] == 6) & (model_3_df["cond_id"] == 2), "condition"] = "trained"
model_3_df.loc[(model_3_df["block_id"] == 6) & (model_3_df["cond_id"] == 1), "condition"] = "extrapolated"

# # first half
# include_condition = (
#     (
#          (cleaned_df["cond_id"].isin([1, 2])) &
#          (cleaned_df["block_id"] == 6) &
#          (cleaned_df["speaker_distance"].isin([3, 4, 5, 6, 7]))
#      ) |
    
#     (
#         (cleaned_df["cond_id"] == 1) &
#         (cleaned_df["block_id"] == 1)
#     )
# )
# filtered_df = cleaned_df[include_condition]

# model_3_df = analysis.get_means_df(filtered_df, "response_distance")

# change condition names
# model_3_df["condition"] = None
# model_3_df.loc[(model_3_df["block_id"] == 1) & (model_3_df["cond_id"] == 1), "condition"] = "naive"
# model_3_df.loc[(model_3_df["block_id"] == 6) & (model_3_df["cond_id"] == 1), "condition"] = "trained"
# model_3_df.loc[(model_3_df["block_id"] == 6) & (model_3_df["cond_id"] == 2), "condition"] = "extrapolated"


# define parameter for the model
x = "speaker_distance"
y= "mean_response_distance"
fixed_group = "condition"
random_group = "sub_id"

# sort reference group for modelling
model_3_df['condition'] = model_3_df['condition'].astype('category')
model_3_df['condition'] = model_3_df['condition'].cat.reorder_categories(["extrapolated", "trained", "naive"], ordered=True)

# log transform data if necessary
model_3_df[f"log_{x}"] = np.log(model_3_df[f"{x}"]) # log transformation doesn't change nonlinearity
model_3_df[f"log_{y}"] = np.log(model_3_df[f"{y}"])
x = f"log_{x}"
y= f"log_{y}"

# ANCOVA with interaction
ANCOVA = smf.ols(formula=f"{y} ~ {x} * C({fixed_group})", data=model_3_df).fit()
print(ANCOVA.summary())

# generate diagnostic plots
cls = LinearRegDiagnostic.LinearRegDiagnostic(ANCOVA)
vif, fig, ax = cls()
print(vif)

# mixed effect ANCOVA with interaction and random slope, intercept
model_3 = smf.mixedlm(
    formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
    groups=model_3_df[f"{random_group}"], # random intercept grouping factor 
    re_formula=f"~{x}", # random slope formula
    data=model_3_df, # data
    ).fit(method=["lbfgs"]) # fitting the model

print(model_3.summary())

# calculare marginal and conditional R2
var_fixed = model_3.fittedvalues.var()
random_effects = model_3.random_effects
var_random = sum([values.var() for values in random_effects.values()])
var_resid = model_3.scale

r2_marginal = var_fixed / (var_fixed + var_random + var_resid) # r2 of fixed effects
r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid) # r2 of fixed and random effects

print(f"Marginal R²: {r2_marginal:.3f}")
print(f"Conditional R²: {r2_conditional:.3f}")

# diagnostic plots for LMM
residuals = model_3.resid
fitted_values = model_3.fittedvalues

# linearity of the predictor
sns.residplot(x=model_3_df[x], y=residuals, lowess=True, line_kws=dict(color="r"))
plt.xlabel(x)
plt.ylabel("Residuals")
plt.title("Residuals vs speaker_distance (linearity)")
plt.show()

# QQ-Plot - normality
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals (normality)")
plt.show()

# Residuals vs fitted - homoscedasticity
sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws=dict(color="r"))
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (homoscedasticity)")
plt.show()

# independency of residuals
plot_acf(residuals)
plt.title("Autocorrelation of Residuals (independency)")
plt.show()

# calculate ICC
resid_var = model_3.scale
group_var = model_3.cov_re.iloc[0, 0]
random_slope_variance = model_3.cov_re.loc[x, x]
ICC = group_var / (group_var + random_slope_variance + resid_var)
print(f"ICC of {random_group}: {ICC:.2f}")

# data visualisation
sns.lmplot(data=model_3_df, x=x, y=y, hue=fixed_group, ci=None, col=fixed_group, order=1)
plt.show()

baseline_x_log = np.log(np.arange(1, 100, 1))
baseline_y_log = np.log(np.arange(1, 100, 1))
sns.lineplot(data=model_3_df, x=x, y=y, hue=fixed_group)
sns.lineplot(x=baseline_x_log, y=baseline_y_log, linestyle="--", color="grey")
plt.xscale("log")
plt.yscale("log")
# plt.axis('equal')
plt.show()



