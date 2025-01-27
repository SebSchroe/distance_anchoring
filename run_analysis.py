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
import statsmodels.api as sm


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
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean=y, mean_by="speaker_distance")

grey_palette = sns.color_palette("grey", n_colors=1)
baseline = np.arange(2, 13, 1)

grid = sns.FacetGrid(data=means_df, col="block_id", row="cond_id", palette="tab10", margin_titles=True)
grid.map_dataframe(sns.lineplot, "speaker_distance", f"mean_{y}", hue="sub_id", estimator=None, lw=1, palette=grey_palette, alpha=0.3)
grid.map_dataframe(sns.pointplot, "speaker_distance", f"mean_{y}", errorbar="sd", native_scale=True, lw=1.5, markersize=5, capsize=0.5, errwidth=1)

for ax in grid.axes.flat:
    ax.plot(baseline, baseline, ls="--", color="black", alpha=0.7) # add 1:1 line through the origin
    
plt.show()

# %% plot individual raw data per participant and save the plot
for sub_id in cleaned_df["sub_id"].unique():
    analysis.plot_data_per_sub(df=cleaned_df, sub_id=sub_id, x="speaker_distance", y="response_distance", baseline="one_one", save=True)

# %% observe questionnaire at its own
analysis.observe_questionnaire(df=questionnaire_df, x="cond_id", y="age", hue="gender")

# %% hypothesis 1  analysis (part I)
# get dataframe as needed
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean="response_distance", mean_by="speaker_distance")
include_condition = (
    ((means_df["cond_id"].isin([1, 2])) &
     (means_df["block_id"] == 1)
     ) |
    ((means_df["cond_id"] == 3) &
     (means_df["block_id"] == 2)
     ))

filtered_df = means_df[include_condition].copy() # filter data by inclusion conditions
filtered_df["block_id"] = filtered_df["block_id"].replace({2: 1}) # set block_id of cond 3 to same level as cond 1 and 2

# remove outliers
filtered_df = filtered_df[filtered_df["sub_id"] != 15] # remove sub 15 as a outlier

model_1_df = filtered_df.copy()

# define parameter for the model
x = "speaker_distance"
y= "mean_response_distance"
fixed_group = "cond_id"
random_group = "sub_id"

# sort reference group for modelling
model_1_df['cond_id'] = model_1_df['cond_id'].astype('category')
model_1_df['cond_id'] = model_1_df['cond_id'].cat.reorder_categories([3, 1, 2], ordered=True)


# log transform data if necessary
model_1_df[f"log_{x}"] = np.log(model_1_df[f"{x}"])
model_1_df[f"log_{y}"] = np.log(model_1_df[f"{y}"])
x = f"log_{x}"
y= f"log_{y}"


for centered_at in [5, 9]:
    
    print(f"\nResults for x centered at {centered_at}:")
    # create temp dataframe
    temp_df = model_1_df.copy()
    # centre x for new intercept intersection
    temp_df["log_centered_speaker_distance"] = temp_df["log_speaker_distance"] - np.log(centered_at)
    x = "log_centered_speaker_distance"
    
    # mixed effect ANCOVA with interaction and random slope, intercept
    model_1 = smf.mixedlm(
        formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
        groups=temp_df[f"{random_group}"], # random intercept grouping factor 
        re_formula="~1", # random intercept for each group in random group
        data=temp_df, # data
        ).fit(method=["powell", "lbfgs"]) # fitting the model
    
    # get all the good analysis stuff
    analysis.LMM_analysis(model_df=model_1_df, fitted_model=model_1, x="log_speaker_distance")

# ANCOVA with interaction
# ANCOVA = smf.ols(formula=f"{y} ~ {x} * C({fixed_group})", data=model_1_df).fit()
# print(ANCOVA.summary())

# # generate diagnostic plots
# cls = LinearRegDiagnostic.LinearRegDiagnostic(ANCOVA)
# vif, fig, ax = cls()
# print(vif)

# R model
# model = lmer(y ~ x * fixed_group categorical + (1 | random_group))

#plot data to visualize hypothesis 1 (part I) results

# extract coefficients from model summary
coefficients = {1: {"k": 0.152 + 0.573, "a": 0.887 - 0.108},
                2: {"k": 0.152 + 1.066, "a": 0.887 - 0.455},
                3: {"k": 0.152, "a": 0.887}}

# create long df with fitted curves
fit_data = []
for cond_id, params in coefficients.items():
    # simulate x_values
    x_values = np.linspace(model_1_df[model_1_df["cond_id"] == cond_id]["speaker_distance"].min(), model_1_df[model_1_df["cond_id"] == cond_id]["speaker_distance"].max(), 100)
    
    # calculate y values
    k = params["k"]
    a = params["a"]
    y_fit = np.exp(k) * x_values**a
    
    # add data to dataframe
    fit_data.extend([(cond_id, x, y) for x, y in zip(x_values, y_fit)])
model_1_fit_df = pd.DataFrame(fit_data, columns=["cond_id", "x_values", "y_values"])

# change order of conditions
model_1_fit_df['cond_id'] = model_1_fit_df['cond_id'].astype('category')
model_1_fit_df['cond_id'] = model_1_fit_df['cond_id'].cat.reorder_categories([3, 1, 2], ordered=True)

# prepare plotting
model_1_df["cond_id"] = model_1_df["cond_id"].map({1: "naive 3-7 m (no view)", 2:"naive 7-11 m (no view)", 3:"naive 2-12 m (limited view)"})
model_1_fit_df["cond_id"] = model_1_fit_df["cond_id"].map({1: "naive 3-7 m (no view)", 2:"naive 7-11 m (no view)", 3:"naive 2-12 m (limited view)"})


baseline = np.arange(2, 13, 1)
ticks = np.arange(2, 13, 1).tolist()
sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.figure(figsize=(6, 6))

# plot data
sns.pointplot(data=model_1_df, x="speaker_distance", y="mean_response_distance", hue="cond_id", errorbar="sd", 
              native_scale=True, ls="None", markersize=5, capsize=5, errwidth=1)
sns.lineplot(data=model_1_fit_df, x="x_values", y="y_values", hue="cond_id", 
             legend=False)
sns.lineplot(x=baseline, y=baseline, linestyle="--", color="grey")

# adjust layout
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log speaker distance [m]")
plt.ylabel("log response distance [m]")
plt.xticks(ticks, labels=[str(tick) for tick in ticks])
plt.yticks(ticks, labels=[str(tick) for tick in ticks])
plt.legend(title="Condition")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.axis("square")
plt.tight_layout()
plt.show()

# %% hypothesis 1 (part II)
# get dataframe
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean="response_distance", mean_by="speaker_distance")
include_condition = (
    ((means_df["cond_id"].isin([1, 2])) &
     (means_df["block_id"].isin([1, 2]))
     )
    )
filtered_df = means_df[include_condition].copy() # filter data by inclusion conditions

# remove outliers
filtered_df = filtered_df[filtered_df["sub_id"] != 15] # remove sub 15 as a outlier

model_1_df = filtered_df.copy()

# define parameter for the model
x = "speaker_distance"
y= "mean_response_distance"
fixed_group = "block_id"
random_group = "sub_id"

# log transform data to fit power function
model_1_df[f"log_{x}"] = np.log(model_1_df[f"{x}"])
model_1_df[f"log_{y}"] = np.log(model_1_df[f"{y}"])
x = f"log_{x}"
y= f"log_{y}"

# mixed effect ANCOVA with interaction and random intercept
for cond_id in [1, 2]:
    temp_df = model_1_df[model_1_df["cond_id"] == cond_id].copy()
    
    # center intercept at certain x
    if cond_id == 1:
        centered_at = 5
    else:
        centered_at = 9
        
    temp_df["log_centered_speaker_distance"] = temp_df["log_speaker_distance"] - np.log(centered_at)    
    x = "log_centered_speaker_distance"
    
    # model
    print(f"\nResults for condition {cond_id} centered at x = {centered_at}")
    model_1 = smf.mixedlm(
        formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
        groups=temp_df[f"{random_group}"], # random intercept grouping factor 
        re_formula="~1", # random intercept for each group in random group
        data=temp_df, # data
        ).fit(method=["powell", "lbfgs"]) # fitting the model
    
    # get all the good analysis stuff    
    analysis.LMM_analysis(model_df=temp_df, fitted_model=model_1, x="log_speaker_distance")
    
# %% plot data for hypothesis 1 (part II)




# %% hypothesis 2 analysis
# set variables
x = "block_id"
y= "absolute_error"
fixed_group = "cond_id"
random_group = "sub_id"

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
               .agg(mean_value=(y, "mean"))
               )

y= f"mean_{y}"
learning_df = learning_df.rename(columns={"mean_value": y})

# exclude block 1
learning_df = learning_df[learning_df["block_id"] != 1]

# data need to be log-transformed to make residuals normal distributed
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
ro.r(f'''
# set independent variables as factor
learning_df$block_id <- as.factor(learning_df$block_id)
learning_df$cond_id <- as.factor(learning_df$cond_id)

# craft the model     
model <- lmer({y} ~ block_id * cond_id + (1|sub_id), data = learning_df)

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
model_2 = smf.mixedlm(
    formula=f"{y} ~ C({x}) * C({fixed_group})", # fixed effect
    groups=learning_df[fixed_group], # random intercept grouping factor 
    re_formula="~1", # random intercept for each group in random group
    data=learning_df, # data
    ).fit(method=["lbfgs"], reml=True) # fitting the model (use reml=False when AIC is needed)
print(model_2.summary())

# calculate marginal, conditional R2 and ICC
var_resid = model_2.scale # var(e)
var_fixed = model_2.fittedvalues.var() # var(f)
var_random = model_2.cov_re.iloc[0, 0]# var(r)

r2_marginal = var_fixed / (var_fixed + var_random + var_resid) # r2 of fixed effects
r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid) # r2 of fixed and random effects
ICC = var_random / (var_random + var_resid) # intraclass correlation coefficient

print(f"Marginal R²: {r2_marginal:.3f}")
print(f"Conditional R²: {r2_conditional:.3f}")
print(f"ICC: {ICC:.3f}")

# diagnostic plots for LMM
sns.reset_orig()
residuals = model_2.resid
fitted_values = model_2.fittedvalues

# linearity of the predictor
sns.residplot(x=learning_df[x], y=residuals, lowess=True, line_kws=dict(color="r"))
plt.xlabel(f"{x}")
plt.ylabel("Residuals")
plt.title(f"Residuals vs {x} (linearity)")
plt.show()

# QQ-Plot - normality
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals (normality)")
plt.show()

# more normality tests
analysis.normality_test(residuals)

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
# prepare visualisation
sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.figure(figsize=(6, 6))
cond_id_mapping = {1: "dist. 3-7 m", 2: "dist. 7-11 m", 3: "dist. 2-12 m"}
block_id_mapping = {1: "naive (no view)", 2: "naive", 4: "1 x trained", 6: "2 x trained"}

sns.catplot(data=learning_df.replace({"cond_id": cond_id_mapping, "block_id": block_id_mapping}),
            x="cond_id", y=y, hue="block_id", 
            kind="box", palette="tab10", legend_out=False)

plt.xlabel(None)
plt.legend(title="Test block")
plt.show()


# %% hypothesis 3 analysis
# set variables
sns.reset_orig()
x = "speaker_distance"
y= "response_distance"
fixed_group = "condition"
random_group = "sub_id"

# create necessary df
# first half speaker_distance
include_condition = (
    (
         (cleaned_df["cond_id"].isin([1, 2])) &
         (cleaned_df["block_id"] == 6) &
         (cleaned_df["speaker_distance"].isin([3, 4, 5, 6, 7]))
     ) |
    
    (
        (cleaned_df["cond_id"] == 1) &
        (cleaned_df["block_id"] == 1)
    )
)
filtered_df = cleaned_df[include_condition] # filter by inclusion conditions
df_1 = analysis.get_means_df(filtered_df, y) # calculate mean values per speaker distance

df_1["condition"] = None
df_1.loc[(df_1["block_id"] == 1) & (df_1["cond_id"] == 1), "condition"] = "naive"
df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 1), "condition"] = "trained"
df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 2), "condition"] = "extrapolated"
df_1["subset"] = "dist. 3-7 m"

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
filtered_df = cleaned_df[include_condition] # filter by inclusion conditions
df_2 = analysis.get_means_df(filtered_df, y) # calculate mean values per speaker distance

# group by condition
df_2["condition"] = None
df_2.loc[(df_2["block_id"] == 1) & (df_2["cond_id"] == 2), "condition"] = "naive"
df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 2), "condition"] = "trained"
df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 1), "condition"] = "extrapolated"
df_2["subset"] = "dist. 7-11 m"

# combine both df's
model_3_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
y = f"mean_{y}"

# sort reference group for modelling
model_3_df['condition'] = model_3_df['condition'].astype('category')
model_3_df['condition'] = model_3_df['condition'].cat.reorder_categories(["naive", "trained", "extrapolated"], ordered=True)

# log transform data if necessary
model_3_df[f"log_{x}"] = np.log(model_3_df[f"{x}"]) # log transformation doesn't change nonlinearity
model_3_df[f"log_{y}"] = np.log(model_3_df[f"{y}"])
x = f"log_{x}"
y= f"log_{y}"

# R: model_3 <- lmer(log_mean_response_distance ~ log_speaker_distance * condition + (1 + log_speaker_distance | sub_id), data = model_3_df)

for subset in model_3_df["subset"].unique():
    
    print(f"\nModel analysis for speaker subset {subset}:")
    subset_df = model_3_df[model_3_df["subset"] == subset]
    
    # mixed effect ANCOVA with interaction and random slope, intercept
    model_3 = smf.mixedlm(
        formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
        groups=subset_df[f"{random_group}"], # random intercept grouping factor 
        re_formula=f"~{x}", # random slope formula
        data=subset_df, # data
        ).fit(method=["lbfgs"]) # fitting the model
    
    print(model_3.summary())
    
    # calculate marginal, conditional R2 and ICC
    var_resid = model_3.scale # var(e)
    var_fixed = model_3.fittedvalues.var() # var(f)
    var_random = model_3.cov_re.iloc[0, 0]# var(r)
    
    r2_marginal = var_fixed / (var_fixed + var_random + var_resid) # r2 of fixed effects
    r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid) # r2 of fixed and random effects
    ICC = var_random / (var_random + var_resid) # intraclass correlation coefficient
    
    print(f"Marginal R²: {r2_marginal:.3f}")
    print(f"Conditional R²: {r2_conditional:.3f}")
    print(f"ICC: {ICC:.3f}")
    
    # diagnostic plots for LMM
    residuals = model_3.resid
    fitted_values = model_3.fittedvalues
    
    # linearity of the predictor
    sns.residplot(x=subset_df[x], y=residuals, lowess=True, line_kws=dict(color="r"))
    plt.xlabel(x)
    plt.ylabel("Residuals")
    plt.title("Residuals vs speaker_distance (linearity)")
    plt.show()
    
    # QQ-Plot - normality
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ-Plot of Residuals (normality)")
    plt.show()
    
    # more normality tests
    analysis.normality_test(residuals)
    
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
    
    # data visualisation
    if subset == "dist. 3-7 m":
        baseline = np.arange(3, 8, 1)
        ticks = np.arange(3, 11, 1).tolist()
    else:
        baseline = np.arange(7, 12, 1)
        ticks = np.arange(6, 12, 1).tolist()
    
    plt.figure(figsize=(8, 8))
    sns.lmplot(data=subset_df, x="speaker_distance", y="mean_response_distance", hue=fixed_group, ci=None, markers="none", legend=False)
    sns.pointplot(data=subset_df, x="speaker_distance", y="mean_response_distance", hue=fixed_group, errorbar="sd",
                  native_scale=True, ls="None", markersize=5, capsize=0.1, errwidth=1, dodge=True)
    sns.lineplot(x=baseline, y=baseline, linestyle="--", color="grey")
    
    plt.xlabel("log speaker distance [m]")
    plt.ylabel("log response distance [m]")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(ticks, labels=[str(tick) for tick in ticks])
    plt.yticks(ticks, labels=[str(tick) for tick in ticks])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.axis("square")
    plt.tight_layout()
    plt.show()

# %% new version hypothesis 3
# set variables
x = "condition"
y= "absolute_error"

# create necessary df
# first half speaker_distance
include_condition = (
    (
         (cleaned_df["cond_id"].isin([1, 2])) &
         (cleaned_df["block_id"] == 6) &
         (cleaned_df["speaker_distance"].isin([3, 4, 5, 6, 7]))
     ) |
    
    (
        (cleaned_df["cond_id"] == 1) &
        (cleaned_df["block_id"] == 1)
    )
)
filtered_df = cleaned_df[include_condition]
df_1 = (filtered_df
               .groupby(["sub_id", "cond_id", "block_id"], as_index=False)
               .agg(mean_value=(y, "mean"))
               )
df_1 = df_1.rename(columns={"mean_value": f"mean_{y}"})

df_1["condition"] = None
df_1.loc[(df_1["block_id"] == 1) & (df_1["cond_id"] == 1), "condition"] = "naive"
df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 1), "condition"] = "trained"
df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 2), "condition"] = "extrapolated"
df_1["subset"] = "dist. 3-7 m"

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

df_2 = (filtered_df
               .groupby(["sub_id", "cond_id", "block_id"], as_index=False)
               .agg(mean_value=(y, "mean"))
               )
df_2 = df_2.rename(columns={"mean_value": f"mean_{y}"})

# group by condition
df_2["condition"] = None
df_2.loc[(df_2["block_id"] == 1) & (df_2["cond_id"] == 2), "condition"] = "naive"
df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 2), "condition"] = "trained"
df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 1), "condition"] = "extrapolated"
df_2["subset"] = "dist. 7-11 m"

# combine both df's
model_3_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
y = f"mean_{y}"

# log transformation
model_3_df[f"log_{y}"] = np.log(model_3_df[y])
y = f"log_{y}"

# sort reference group for modelling
model_3_df['condition'] = model_3_df['condition'].astype('category')
model_3_df['condition'] = model_3_df['condition'].cat.reorder_categories(["naive", "trained", "extrapolated"], ordered=True)


# model each speaker subset seperately
formula = f"{y} ~ C({x})"
for subset in model_3_df["subset"].unique():
    
    print(f"\nModel analysis for speaker subset {subset}:")
    
    subset_df = model_3_df[model_3_df["subset"] == subset]
    # create model
    ANOVA = smf.ols(formula=formula, data=subset_df).fit()
    anova_table = sm.stats.anova_lm(ANOVA, typ=2)
    print(anova_table)
    r2 = ANOVA.rsquared
    print(f"r2: {r2:.3f}")

    # check assuptions
    cls = LinearRegDiagnostic.LinearRegDiagnostic(ANOVA)
    vif, fig, ax = cls()
    # print(vif)
    
    # more normality tests
    residuals = ANOVA.resid
    analysis.normality_test(residuals)

    # tukey hsd post hoc test
    tukey_result = pairwise_tukeyhsd(endog=subset_df[y], groups=subset_df[x])
    print("\n", tukey_result)

# visualize data
sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
sns.catplot(data=model_3_df, x=x, y=y, col="subset", hue="cond_id", kind="box", palette="tab10")




