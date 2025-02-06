# %% prepare data
# import modules
import analysis
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import LinearRegDiagnostic
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from scipy.stats import page_trend_test


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
analysis.observe_questionnaire(df=questionnaire_df, x="cond_id", y="q09", hue=None)

# %% hypothesis 1  analysis (part I)
sns.reset_orig()

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
filtered_df = filtered_df[filtered_df["sub_id"] != 15] # remove sub 15 as extreme outlier in block 1

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
model_1_df["cond_id"] = model_1_df["cond_id"].map({1: "3-7 m (no view)", 2:"7-11 m (no view)", 3:"2-12 m (limited view)"})
model_1_fit_df["cond_id"] = model_1_fit_df["cond_id"].map({1: "3-7 m (no view)", 2:"7-11 m (no view)", 3:"2-12 m (limited view)"})


baseline = np.arange(2, 13, 1)
hue_order = ("3-7 m (no view)", "7-11 m (no view)", "2-12 m (limited view)")
ticks = np.arange(2, 13, 1).tolist()
sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.figure(figsize=(6, 6))

# plot data
sns.pointplot(data=model_1_df, x="speaker_distance", y="mean_response_distance", hue="cond_id", errorbar="sd", 
              native_scale=True, ls="None", markersize=5, capsize=5, errwidth=1, hue_order=hue_order)
sns.lineplot(data=model_1_fit_df, x="x_values", y="y_values", hue="cond_id", 
             legend=False, hue_order=hue_order)
sns.lineplot(x=baseline, y=baseline, linestyle="--", color="grey")
plt.axvline(x=5, ymin=0, ymax=1, color="grey", alpha=0.7)
plt.axvline(x=9, ymin=0, ymax=1, color="grey", alpha=0.7)


# adjust layout
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log speaker distance [m]")
plt.ylabel("log response distance [m]")
plt.xticks(ticks, labels=[str(tick) for tick in ticks])
plt.yticks(ticks, labels=[str(tick) for tick in ticks])
plt.legend(title="Naive condition")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.axis("square")
plt.tight_layout()
plt.show()

# %% hypothesis 1 (part II)
sns.reset_orig()

# get dataframe
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean="response_distance", mean_by="speaker_distance")
include_condition = (
    ((means_df["cond_id"].isin([1, 2])) &
     (means_df["block_id"].isin([1, 2]))
     )
    )
filtered_df = means_df[include_condition].copy() # filter data by inclusion conditions

# remove outliers
filtered_df = filtered_df[filtered_df["sub_id"] != 15] # remove sub 15 as extreme outlier in block 1

# prepare analysis
model_1_df = filtered_df.copy()
sns.lmplot(data=model_1_df, x="speaker_distance", y="mean_response_distance", hue="block_id", row="cond_id", ci=None)
plt.show()

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
fit_data = []
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
    
    # get coefficients
    # help model
    x = "log_speaker_distance"
    help_model = smf.mixedlm(
        formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
        groups=temp_df[f"{random_group}"], # random intercept grouping factor 
        re_formula="~1", # random intercept for each group in random group
        data=temp_df, # data
        ).fit(method=["powell", "lbfgs"]) # fitting the model
    
    intercept_1 = help_model.params[0]
    intercept_2 = intercept_1 + help_model.params[1]
    slope_1 = help_model.params[2]
    slope_2 = slope_1 + help_model.params[3]
    
    coefficients = {1: {"k": intercept_1, "a": slope_1},
                    2: {"k": intercept_2, "a": slope_2}}    
    
    # calculate fitting line
    for block_id, params in coefficients.items():
        # simulate x_values
        min_x = model_1_df[(model_1_df["cond_id"] == cond_id) & (model_1_df["block_id"] == block_id)]["speaker_distance"].min()
        max_x = model_1_df[(model_1_df["cond_id"] == cond_id) & (model_1_df["block_id"] == block_id)]["speaker_distance"].max()
        x_values = np.linspace(min_x, max_x, 100)
        
        # calculate y values
        k = params["k"]
        a = params["a"]
        y_fit = np.exp(k) * x_values**a
        
        # add data to dataframe
        fit_data.extend([(cond_id, block_id, x, y) for x, y in zip(x_values, y_fit)])
    
    
    
# %% plot data for hypothesis 1 (part II)
# get fitted dataframe and data
model_1_fit_df = pd.DataFrame(fit_data, columns=["cond_id", "block_id", "x_values", "y_values"])
cond_ids = model_1_df["cond_id"].unique()
block_ids = model_1_df["block_id"].unique()

#prepare plotting
sns.reset_orig()
sns.set_theme(style="whitegrid", context="notebook")

palette = sns.color_palette("colorblind", n_colors=len(cond_ids))

# create subplots
fig, axes = plt.subplots(1, 2, sharey=True)

# create every plot separately
for ax, cond_id in zip(axes, cond_ids):
    
    # plot average data with with std
    sns.pointplot(data=model_1_df[model_1_df["cond_id"] == cond_id], 
                  x="speaker_distance", y="mean_response_distance", hue="block_id",
                  errorbar="sd", ls="None", native_scale=True, ax=ax,
                  markersize=5, capsize=0.2, errwidth=1,
                  palette=palette)
    
    # plot fitted lines
    for block_id in block_ids:
        subset = model_1_fit_df[(model_1_fit_df["cond_id"] == cond_id) & (model_1_fit_df["block_id"] == block_id)]
        ax.plot(subset["x_values"], subset["y_values"])
        
    # plot baseline
    baseline = np.linspace(model_1_df[model_1_df["cond_id"] == cond_id]["speaker_distance"].min(), model_1_df[model_1_df["cond_id"] == cond_id]["speaker_distance"].max(), 100)
    ax.plot(baseline, baseline, ls="--", color="black", alpha=0.7)
    
    # layout
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("log speaker distance [m]")
    ax.set_ylabel("log response distance [m]")

plt.show()


# %% hypothesis 2 analysis
# # set variables
# x = "block_id"
# y= "absolute_error"
# fixed_group = "cond_id"
# random_group = "sub_id"

# # get dataframe as it is needed (for block 6 only speaker of speaker subset used)
# remove_condition = (
#     (
#         (cleaned_df["block_id"] == 6) &
#         (cleaned_df["cond_id"] == 1) &
#         (cleaned_df["speaker_distance"].isin([2, 8, 9, 10, 11, 12]))
#     ) |
#     (
#         (cleaned_df["block_id"] == 6) &
#         (cleaned_df["cond_id"] == 2) &
#         (cleaned_df["speaker_distance"].isin([2, 3, 4, 5, 6, 12]))
#     )
# )
# filtered_df = cleaned_df[~remove_condition].copy()
# filtered_df = filtered_df[filtered_df["block_id"] != 1]
# # filtered_df = filtered_df[~filtered_df["sub_id"].isin([6])] # remove extreme outliers

# means_df = analysis.get_means_df(df=filtered_df, value_to_mean=y, mean_by="block_id")

# # prepare df for modeling
# model_2_df = means_df.copy()

# # check data
# sns.lmplot(data=means_df, x="block_id", y="mean_absolute_error", hue="sub_id", row="cond_id", ci=None)

# # data need to be log-transformed to make residuals normal distributed
# model_2_df[f"log_{y}"] = np.log(model_2_df[f"mean_{y}"])
# y = f"log_{y}"

# # set parameters as categorical
# # learning_df["block_id"] = learning_df["block_id"].astype("category")
# # learning_df["cond_id"] = learning_df["cond_id"].astype("category")
# # learning_df["sub_id"] = learning_df["sub_id"].astype("category")

# # transfer pandas df to R
# ro.globalenv["model_2_df"] = pandas2ri.py2rpy(model_2_df)
# print(ro.r("head(model_2_df)")) # check dataframe

# # mixed effect two way ANOVA with ineraction in R
# ro.r(f'''
# # set independent variables as factor
# model_2_df$block_id <- as.factor(model_2_df$block_id)
# model_2_df$cond_id <- as.factor(model_2_df$cond_id)

# # craft the model     
# model <- lmer({y} ~ block_id * cond_id + (1|sub_id), data = model_2_df)

# # tukey hsd post hoc test
# tukey <- emmeans(model, pairwise ~ block_id * cond_id, adjust = "tukey")
# ''')

# # extrakt results tables from R
# print("\nMixed effect ANOVA table:")
# print(ro.r("anova(model)"))
# print("\nMixed effect ANOVA summary table:")
# print(ro.r("summary(model)"))
# print("\nTukey HSD post-hoc-test table:")
# print(ro.r("summary(tukey)"))


# # mixed effects ANOVA with interaction
# model_2 = smf.mixedlm(
#     formula=f"{y} ~ C({x}) * C({fixed_group})", # fixed effect
#     groups=model_2_df[random_group], # random intercept grouping factor 
#     re_formula="~1", # random intercept for each group in random group
#     data=model_2_df, # data
#     ).fit(method=["powell", "lbfgs"], reml=True) # fitting the model (use reml=False when AIC is needed)

# analysis.LMM_analysis(model_df=model_2_df, fitted_model=model_2, x="block_id")

# # visualise data
# # prepare visualisation
# sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
# plt.figure(figsize=(6, 6))
# cond_id_mapping = {1: "dist. 3-7 m", 2: "dist. 7-11 m", 3: "dist. 2-12 m"}
# block_id_mapping = {1: "naive (no view)", 2: "naive", 4: "1 x trained", 6: "2 x trained"}

# sns.catplot(data=model_2_df.replace({"cond_id": cond_id_mapping, "block_id": block_id_mapping}),
#             x="cond_id", y=y, hue="block_id", 
#             kind="box", palette="tab10", legend_out=False)

# plt.xlabel(None)
# plt.legend(title="Test block")
# plt.show()

# %% hypothesis 2 page's L test
# get dataframe as it is needed (for block 6 only speaker of speaker subset used)
sns.reset_orig()

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
filtered_df = cleaned_df[~remove_condition].copy()
filtered_df = filtered_df[filtered_df["block_id"] != 1]
# filtered_df = filtered_df[~filtered_df["sub_id"].isin([13, 15])] # no outlier removal because we want to investigate the effect of training

means_df = analysis.get_means_df(df=filtered_df, value_to_mean="absolute_error", mean_by="block_id")

# prepare df for modeling
model_2_df = means_df.copy()

for cond_id in model_2_df["cond_id"].unique():
    
    # filter by cond_id
    temp_df = model_2_df[model_2_df["cond_id"] == cond_id]
    
    # pivot df and transform to numpy array
    pivot_df = temp_df.pivot(index="sub_id", columns="block_id", values="mean_absolute_error")
    data = pivot_df.iloc[:, ::-1].to_numpy() # reverse block_ids to test for monotonously decreasing error
    
    # page's L test
    results = page_trend_test(data=data, method="exact")
    
    print(f"\nResults of Page's L test for condition: {cond_id}")
    print(results)
    
# visualise data
# prepare visualisation
sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.figure(figsize=(6, 6))
cond_id_mapping = {1: "dist. 3-7 m", 2: "dist. 7-11 m", 3: "dist. 2-12 m"}
block_id_mapping = {2: "untrained", 4: "single-trained", 6: "double-trained"}

sns.catplot(data=model_2_df.replace({"cond_id": cond_id_mapping, "block_id": block_id_mapping}),
            x="cond_id", y="mean_absolute_error", hue="block_id", 
            kind="box", palette="tab10", legend_out=False)

plt.xlabel(None)
plt.legend(title="Stage")
plt.show()
    
# %% hypothesis 3 (part I)
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
         (cleaned_df["speaker_distance"].isin([2, 3, 4, 5, 6]))
     )
)
filtered_df = cleaned_df[include_condition] # filter by inclusion conditions
df_1 = analysis.get_means_df(df=filtered_df, value_to_mean=y, mean_by=x) # calculate mean values per speaker distance

df_1["condition"] = None
df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 1), "condition"] = "trained"
df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 2), "condition"] = "generalised"
df_1["subset"] = "dist. 2-6 m"

# second half
include_condition = (
    (
         (cleaned_df["cond_id"].isin([1, 2])) &
         (cleaned_df["block_id"] == 6) &
         (cleaned_df["speaker_distance"].isin([8, 9, 10, 11, 12]))
     )
)
filtered_df = cleaned_df[include_condition] # filter by inclusion conditions
df_2 = analysis.get_means_df(df=filtered_df, value_to_mean=y, mean_by=x) # calculate mean values per speaker distance

# group by condition
df_2["condition"] = None
df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 2), "condition"] = "trained"
df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 1), "condition"] = "generalised"
df_2["subset"] = "dist. 8-12 m"

# combine both df's
filtered_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
y = f"mean_{y}"

# sort reference group for modelling
filtered_df['condition'] = filtered_df['condition'].astype('category')
filtered_df['condition'] = filtered_df['condition'].cat.reorder_categories(["generalised", "trained"], ordered=True)

# remove some data
# filtered_df = filtered_df[filtered_df["condition"] != "naive"]
filtered_df = filtered_df[filtered_df["sub_id"] != 8]

# prepare analysis
model_3_df = filtered_df.copy()
sns.lmplot(data=model_3_df, x="speaker_distance", y="mean_response_distance", hue="sub_id", col="condition", ci=False)
plt.show()

# log transform data if necessary
model_3_df[f"log_{x}"] = np.log(model_3_df[f"{x}"]) # log transformation doesn't change nonlinearity
model_3_df[f"log_{y}"] = np.log(model_3_df[f"{y}"])
x = f"log_{x}"
y= f"log_{y}"

# R: model_3 <- lmer(log_mean_response_distance ~ log_speaker_distance * condition + (1 | sub_id), data = model_3_df)

for subset in model_3_df["subset"].unique():
    
    temp_df = model_3_df[model_3_df["subset"] == subset]
    
    # center intercept at certain x
    if subset == "dist. 2-6 m":
        centered_at = 4
    else:
        centered_at = 10
        
    temp_df["log_centered_speaker_distance"] = temp_df["log_speaker_distance"] - np.log(centered_at)
    x = "log_centered_speaker_distance"
    
    print(f"\nModel analysis for speaker subset {subset} centered at {centered_at}:")
    
    # mixed effect ANCOVA with interaction and random slope, intercept
    model_3 = smf.mixedlm(
        formula=f"{y} ~ {x} * C({fixed_group})", # fixed effect
        groups=temp_df[f"{random_group}"], # random intercept grouping factor 
        re_formula="~1", # only random intercept
        data=temp_df, # data
        ).fit(method=["powell", "lbfgs"]) # fitting the model
    
    # get all the good analysis stuff    
    analysis.LMM_analysis(model_df=temp_df, fitted_model=model_3, x=x)
    
    # data visualisation
    if subset == "dist. 2-6 m":
        baseline = np.arange(2, 7, 1)
        ticks = np.arange(2, 8, 1).tolist()
    else:
        baseline = np.arange(8, 13, 1)
        ticks = np.arange(6, 13, 1).tolist()
    
    plt.figure(figsize=(8, 8))
    sns.lmplot(data=temp_df, x="speaker_distance", y="mean_response_distance", hue=fixed_group, ci=None, markers="none", legend=False)
    sns.pointplot(data=temp_df, x="speaker_distance", y="mean_response_distance", hue=fixed_group, errorbar="sd",
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

# %% hypothesis 3 (part II)
# absolute error per speaker distance in block 6
# divide in learned and generalised
# page L test in both "zones"

# create necessary df
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean="absolute_error", mean_by="speaker_distance")

include_condition = (
    (
     (means_df["cond_id"].isin([1, 2])) &
     (means_df["block_id"] == 6)
     )
    )
filtered_df = means_df[include_condition].copy()

filtered_df["condition"] = None
filtered_df.loc[(filtered_df["cond_id"] == 1) & (filtered_df["speaker_distance"].isin([3, 4, 5, 6, 7])), "condition"] = "trained 3-7 m"
filtered_df.loc[(filtered_df["cond_id"] == 1) & (filtered_df["speaker_distance"].isin([8, 9, 10, 11, 12])), "condition"] = "generalised 8-12 m"

filtered_df.loc[(filtered_df["cond_id"] == 2) & (filtered_df["speaker_distance"].isin([7, 8, 9, 10, 11])), "condition"] = "trained 7-11 m"
filtered_df.loc[(filtered_df["cond_id"] == 2) & (filtered_df["speaker_distance"].isin([2, 3, 4, 5, 6])), "condition"] = "generalised 2-6 m"

# prepare analysis
model_3_df = filtered_df.copy()
model_3_df = model_3_df.dropna()
sns.lmplot(data=model_3_df, x="speaker_distance", y="mean_absolute_error", hue="sub_id", col="condition", ci=None)

for trend in ["increasing", "decreasing"]:
    
    print(f"\nResults for {trend} trends:")
    for condition in model_3_df["condition"].unique():
        
        # filter by condition
        temp_df = model_3_df[model_3_df["condition"] == condition]
        
        # pivot df and transform to numpy array
        pivot_df = temp_df.pivot(index="sub_id", columns="speaker_distance", values="mean_absolute_error")
        
        # reverse speaker distance depending on condition (measure from middle)
        if trend == "increasing":
            data = pivot_df.to_numpy()
        else:
            data = pivot_df.iloc[:, ::-1].to_numpy() # reverse speaker distance to check for monotoneously increase from known distances 
        
        # page's L test
        results = page_trend_test(data=data, method="exact")
        
        print(f"Results of Page's L test for condition: {condition}")
        print(results)

# %% new version hypothesis 3
# # set variables
# x = "condition"
# y= "absolute_error"

# # create necessary df
# # first half speaker_distance
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
# df_1 = (filtered_df
#                .groupby(["sub_id", "cond_id", "block_id"], as_index=False)
#                .agg(mean_value=(y, "mean"))
#                )
# df_1 = df_1.rename(columns={"mean_value": f"mean_{y}"})

# df_1["condition"] = None
# df_1.loc[(df_1["block_id"] == 1) & (df_1["cond_id"] == 1), "condition"] = "naive"
# df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 1), "condition"] = "trained"
# df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 2), "condition"] = "extrapolated"
# df_1["subset"] = "dist. 3-7 m"

# # second half
# include_condition = (
#     (
#          (cleaned_df["cond_id"].isin([1, 2])) &
#          (cleaned_df["block_id"] == 6) &
#          (cleaned_df["speaker_distance"].isin([7, 8, 9, 10, 11]))
#      ) |
    
#     (
#         (cleaned_df["cond_id"] == 2) &
#         (cleaned_df["block_id"] == 1)
#     )
# )
# filtered_df = cleaned_df[include_condition]

# df_2 = (filtered_df
#                .groupby(["sub_id", "cond_id", "block_id"], as_index=False)
#                .agg(mean_value=(y, "mean"))
#                )
# df_2 = df_2.rename(columns={"mean_value": f"mean_{y}"})

# # group by condition
# df_2["condition"] = None
# df_2.loc[(df_2["block_id"] == 1) & (df_2["cond_id"] == 2), "condition"] = "naive"
# df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 2), "condition"] = "trained"
# df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 1), "condition"] = "extrapolated"
# df_2["subset"] = "dist. 7-11 m"

# # combine both df's
# model_3_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
# y = f"mean_{y}"

# # log transformation
# model_3_df[f"log_{y}"] = np.log(model_3_df[y])
# y = f"log_{y}"

# # sort reference group for modelling
# model_3_df['condition'] = model_3_df['condition'].astype('category')
# model_3_df['condition'] = model_3_df['condition'].cat.reorder_categories(["naive", "trained", "extrapolated"], ordered=True)


# # model each speaker subset seperately
# formula = f"{y} ~ C({x})"
# for subset in model_3_df["subset"].unique():
    
#     print(f"\nModel analysis for speaker subset {subset}:")
    
#     subset_df = model_3_df[model_3_df["subset"] == subset]
#     # create model
#     ANOVA = smf.ols(formula=formula, data=subset_df).fit()
#     anova_table = sm.stats.anova_lm(ANOVA, typ=2)
#     print(anova_table)
#     r2 = ANOVA.rsquared
#     print(f"r2: {r2:.3f}")

#     # check assuptions
#     cls = LinearRegDiagnostic.LinearRegDiagnostic(ANOVA)
#     vif, fig, ax = cls()
#     # print(vif)
    
#     # more normality tests
#     residuals = ANOVA.resid
#     analysis.normality_test(residuals)

#     # tukey hsd post hoc test
#     tukey_result = pairwise_tukeyhsd(endog=subset_df[y], groups=subset_df[x])
#     print("\n", tukey_result)

# # visualize data
# sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
# sns.catplot(data=model_3_df, x=x, y=y, col="subset", hue="cond_id", kind="box", palette="tab10")
