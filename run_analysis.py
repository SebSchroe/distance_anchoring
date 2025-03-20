# %% prepare data
# import modules
import analysis
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import page_trend_test, ttest_ind
from statannotations.Annotator import Annotator

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
# set theme and style
analysis.define_plotting_theme()

y = "response_distance"
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean=y, mean_by="speaker_distance")
means_df["block_id"] = means_df["block_id"].map({1: "naive (no view)", 2:"naive (limited view)", 4:"single-trained (limited view)", 6: "double-trained (limited view)"})
means_df["cond_id"] = means_df["cond_id"].map({1: "Group A (trained 3-7 m)", 2:"Group B (trained 7-11 m)", 3:"Reference (trained 2-12 m)"})

grey_palette = sns.color_palette("grey", n_colors=1)
baseline = np.arange(2, 13, 1)

grid = sns.FacetGrid(data=means_df, col="block_id", row="cond_id", palette="tab10", margin_titles=True)
grid.map_dataframe(sns.lineplot, "speaker_distance", f"mean_{y}", hue="sub_id", estimator=None, lw=1, palette=grey_palette, alpha=0.3)
grid.map_dataframe(sns.pointplot, "speaker_distance", f"mean_{y}", errorbar="sd", native_scale=True, lw=1.5, markersize=5, capsize=0.5, errwidth=1)

for ax in grid.axes.flat:
    ax.plot(baseline, baseline, ls="--", color="black", alpha=0.7) # add 1:1 line through the origin

grid.set_axis_labels(x_var="stimulus distance [m]", y_var="estimated distance [m]")
grid.set_titles(col_template="{col_name}", row_template="{row_name}", weight="bold")

grid.tight_layout()
analysis.save_fig(name="all_data") # save plot under figure as jpg with 300 dpi

# %% plot individual raw data per participant and save the plot
for sub_id in cleaned_df["sub_id"].unique():
    analysis.plot_data_per_sub(df=cleaned_df, sub_id=sub_id, x="speaker_distance", y="response_distance", baseline="one_one", save=True)

# %% observe questionnaire at its own
analysis.observe_questionnaire(df=questionnaire_df, x="cond_id", y="q09", hue=None)
    
# %% Hypothesis 1 (part I)
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
filtered_df = filtered_df[filtered_df["sub_id"] != 15] # remove sub 15 as extreme outlier in block 1

model_1_df = filtered_df.copy()

# sort reference group for modelling
model_1_df['cond_id'] = model_1_df['cond_id'].astype('category')
model_1_df['cond_id'] = model_1_df['cond_id'].cat.reorder_categories([3, 1, 2], ordered=True)

# run model plus analysis
analysis.run_mixedlm_analysis(df=model_1_df, x="speaker_distance", y="mean_response_distance", fixed_group="cond_id", random_group="sub_id", centered_at_values=[5, 9])

# %% Visualisation hypothesis 1 (part I)
# set theme and style
analysis.define_plotting_theme()

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
model_1_df["cond_id"] = model_1_df["cond_id"].map({1: "Group A (naive, no view)", 2:"Group B (naive, no view)", 3:"Reference (naive, limited view)"})
model_1_fit_df["cond_id"] = model_1_fit_df["cond_id"].map({1: "Group A (naive, no view)", 2:"Group B (naive, no view)", 3:"Reference (naive, limited view)"})

baseline = np.arange(2, 13, 1)
hue_order = ("Group A (naive, no view)", "Group B (naive, no view)", "Reference (naive, limited view)")
ticks = np.arange(2, 13, 1).tolist()
plt.figure(figsize=(6, 6))

# plot data
sns.pointplot(data=model_1_df, x="speaker_distance", y="mean_response_distance", hue="cond_id", errorbar="sd", 
              native_scale=True, ls="None", markersize=5, capsize=5, errwidth=1, hue_order=hue_order, palette="colorblind")
sns.lineplot(data=model_1_fit_df, x="x_values", y="y_values", hue="cond_id", 
             legend=False, hue_order=hue_order, palette="colorblind")
sns.lineplot(x=baseline, y=baseline, linestyle="--", color="grey")
plt.axvline(x=5, ymin=0, ymax=1, color="grey")
plt.axvline(x=9, ymin=0, ymax=1, color="grey")

# adjust layout
plt.xscale("log")
plt.yscale("log")
plt.xlabel("log stimulus distance [m]")
plt.ylabel("log estimated distance [m]")
plt.xticks(ticks, labels=[str(tick) for tick in ticks])
plt.yticks(ticks, labels=[str(tick) for tick in ticks])
plt.legend(title="Group (Condition)", loc="center left", bbox_to_anchor=(1, 0.5), alignment="center", handletextpad=0, frameon=False)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.axis("square")
plt.tight_layout()
analysis.save_fig("hyp-1_part-1")

# %% Hypothesis 1 (part II)
# get dataframe
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean="response_distance", mean_by="speaker_distance")
include_condition = (
    ((means_df["cond_id"].isin([1, 2])) &
     (means_df["block_id"].isin([1, 2]))
     )
    )
filtered_df = means_df[include_condition].copy() # filter data by inclusion conditions
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
    
# %% Visualisation hypothesis 1 (part II)
# get fitted dataframe and data
model_1_fit_df = pd.DataFrame(fit_data, columns=["cond_id", "block_id", "x_values", "y_values"])

# rename conditions
model_1_df["cond_id"] = model_1_df["cond_id"].replace({1: "3-7 m", 2:"7-11 m"})
model_1_fit_df["cond_id"] = model_1_fit_df["cond_id"].replace({1: "3-7 m", 2:"7-11 m"})

model_1_df["block_id"] = model_1_df["block_id"].replace({1: "naive, no view", 2:"naive, limited view"})
model_1_fit_df["block_id"] = model_1_fit_df["block_id"].replace({1: "naive, no view", 2:"naive, limited view"})

# prepare looping
cond_ids = model_1_df["cond_id"].unique().tolist()
block_ids = model_1_df["block_id"].unique().tolist()

#prepare plotting
# set theme and style
analysis.define_plotting_theme()

y_ticks = np.arange(3, 12, 1).tolist()

# create subplots
fig, axes = plt.subplots(1, 2, sharey=True)

# create every plot separately
for ax, cond_id in zip(axes, cond_ids):
    
    # plot average data with with std
    sns.pointplot(data=model_1_df[model_1_df["cond_id"] == cond_id], 
                  x="speaker_distance", y="mean_response_distance", hue="block_id",
                  errorbar="sd", ls="None", native_scale=True, ax=ax,
                  markersize=5, capsize=0.2, errwidth=1,
                  palette="colorblind")
    
    # plot fitted lines
    for block_id in block_ids:
        subset = model_1_fit_df[(model_1_fit_df["cond_id"] == cond_id) & (model_1_fit_df["block_id"] == block_id)]
        ax.plot(subset["x_values"], subset["y_values"])
        x_ticks = np.arange(subset["x_values"].min(), subset["x_values"].max() + 1, 1).astype(int).tolist()
        
    # plot baseline
    baseline = np.linspace(model_1_df[model_1_df["cond_id"] == cond_id]["speaker_distance"].min(), model_1_df[model_1_df["cond_id"] == cond_id]["speaker_distance"].max(), 100)
    ax.plot(baseline, baseline, ls="--", color="black", alpha=0.7)
    
    # layout
    ax.set_title(cond_id)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("log stimulus distance [m]")
    ax.set_ylabel("log estimated distance [m]")
    ax.set_yticks(y_ticks, labels=[str(y_tick) for y_tick in y_ticks])
    ax.set_xticks(x_ticks, labels=[str(x_tick) for x_tick in x_ticks])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    
fig.legend(handles, labels, title="Test", loc="center left", bbox_to_anchor=(0.98, 0.5), handletextpad=0, frameon=False)

plt.tight_layout()
analysis.save_fig("hyp-1_part-2")

# %% Hypothesis 1 (part III)
# t-test comparison between block 1 and 2 of cond 1 and 2
means_df = analysis.get_means_df(df=cleaned_df, value_to_mean="absolute_error", mean_by="speaker_distance")
include_condition = (
    ((means_df["cond_id"].isin([1, 2])) &
     (means_df["block_id"].isin([1, 2]))
     )
    )
filtered_df = means_df[include_condition].copy()
filtered_df = filtered_df[filtered_df["sub_id"] != 15]
sns.lmplot(data=filtered_df, x="block_id", y="mean_absolute_error", col="cond_id", hue="cond_id")
plt.show()

group_1_array = filtered_df[(filtered_df["cond_id"] == 2) & (filtered_df["block_id"] == 1)]["mean_absolute_error"].to_numpy()
group_2_array = filtered_df[(filtered_df["cond_id"] == 2) & (filtered_df["block_id"] == 2)]["mean_absolute_error"].to_numpy()

t_stat, p_val = ttest_ind(a=group_1_array, b=group_2_array, equal_var=False, alternative="two-sided")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.3f}")

group_1_parameter = analysis.get_group_parameter(array=group_1_array)
group_2_parameter = analysis.get_group_parameter(array=group_2_array)
analysis.statistical_power(group_1=group_1_parameter, group_2=group_2_parameter, alpha=0.05, alternative="two-sided")

# %% Hypothesis 1 (part IV)
# t-test questionaire
group_1_array = questionnaire_df[questionnaire_df["cond_id"] == 1]["q09"].to_numpy()
group_2_array = questionnaire_df[questionnaire_df["cond_id"] == 2]["q09"].to_numpy()

group_1_array = np.delete(group_1_array, obj=3)

t_stat, p_val = ttest_ind(a=group_1_array, b=group_2_array, equal_var=False, alternative="two-sided")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.3f}")

group_1_parameter = analysis.get_group_parameter(array=group_1_array)
group_2_parameter = analysis.get_group_parameter(array=group_2_array)
analysis.statistical_power(group_1=group_1_parameter, group_2=group_2_parameter, alpha=0.05, alternative="two-sided")

# %% Hypothesis 2 (part I)
# page's L test
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
filtered_df = cleaned_df[~remove_condition].copy()
filtered_df = filtered_df[filtered_df["block_id"] != 1]

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

# %% Hypothesis 2 (part II)
# t-test comparison accuracy block 2 cond 1, 2 and 3
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
filtered_df = filtered_df[filtered_df["block_id"].isin([2, 6])]

means_df = analysis.get_means_df(df=filtered_df, value_to_mean="absolute_error", mean_by="block_id")

cond_to_compare = [1, 3]
group_1_array = means_df[(means_df["block_id"] == 2) & (means_df["cond_id"] == cond_to_compare[0])]["mean_absolute_error"].to_numpy()
group_2_array = means_df[(means_df["block_id"] == 2) & (means_df["cond_id"] == cond_to_compare[1])]["mean_absolute_error"].to_numpy()

t_stat, p_val = ttest_ind(a=group_1_array, b=group_2_array, equal_var=False, alternative="two-sided")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.3f}")


group_1_parameter = analysis.get_group_parameter(array=group_1_array)
group_2_parameter = analysis.get_group_parameter(array=group_2_array)
analysis.statistical_power(group_1=group_1_parameter, group_2=group_2_parameter, alpha=0.05, alternative="two-sided")

# %% Visualisation hypothesis II (part II)

# get dataframe and map cond_id / block_id
annotations_df = means_df.copy()
cond_id_mapping = {1: "Group A (3-7 m)", 2: "Group B (7-11 m)", 3: "Reference (2-12 m)"}
block_id_mapping = {2: "naive (limited view)", 6: "double-trained (limited view)"}
annotations_df = annotations_df.replace({"cond_id": cond_id_mapping, "block_id": block_id_mapping})

x = "block_id"
y = "mean_absolute_error"
cond_ids = [1, 2, 3]

pairs = [
    [("naive (limited view)", "Group A (3-7 m)"), ("naive (limited view)", "Group B (7-11 m)")],
    [("naive (limited view)", "Group A (3-7 m)"), ("naive (limited view)", "Reference (2-12 m)")],
    [("naive (limited view)", "Group B (7-11 m)"), ("naive (limited view)", "Reference (2-12 m)")],
    
    [("double-trained (limited view)", "Group A (3-7 m)"), ("double-trained (limited view)", "Group B (7-11 m)")],
    [("double-trained (limited view)", "Group A (3-7 m)"), ("double-trained (limited view)", "Reference (2-12 m)")],
    [("double-trained (limited view)", "Group B (7-11 m)"), ("double-trained (limited view)", "Reference (2-12 m)")]
    ]

hue_plot_params = {
    'data': annotations_df,
    'x': 'block_id',
    'y': 'mean_absolute_error',
    "order": ["naive (limited view)", "double-trained (limited view)"],
    "hue": "cond_id",
    "hue_order": ["Group A (3-7 m)", "Group B (7-11 m)", "Reference (2-12 m)"],
    "palette": "colorblind"
}

# set theme and style
analysis.define_plotting_theme()

# Plot with seaborn
ax = sns.boxplot(**hue_plot_params)

# Add annotations
annotator = Annotator(ax, pairs, **hue_plot_params)
annotator.configure(test="Mann-Whitney").apply_and_annotate()

# adjust layout
plt.xlabel(None)
plt.ylabel("mean absolute error")
plt.legend(title="Group", loc="center left", bbox_to_anchor=(1, 0.5), alignment="center", handletextpad=0.5, frameon=False)

#plt.tight_layout()
analysis.save_fig("hyp-2_part-2")
    
# %% Hypothesis 3 (part I)
# set variables
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
df_1.loc[(df_1["block_id"] == 6) & (df_1["cond_id"] == 2), "condition"] = "novel"
df_1["subset"] = "2-6 m"

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
df_2.loc[(df_2["block_id"] == 6) & (df_2["cond_id"] == 1), "condition"] = "novel"
df_2["subset"] = "8-12 m"

# combine both df's
filtered_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
y = f"mean_{y}"

# sort reference group for modelling
filtered_df['condition'] = filtered_df['condition'].astype('category')
filtered_df['condition'] = filtered_df['condition'].cat.reorder_categories(["trained", "novel"], ordered=True)

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

fit_data = []
for subset in model_3_df["subset"].unique():
    
    temp_df = model_3_df[model_3_df["subset"] == subset]
    
    # center intercept at certain x
    if subset == "2-6 m":
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
    
    coefficients = {"trained": {"k": intercept_1, "a": slope_1},
                    "novel": {"k": intercept_2, "a": slope_2}}    
    
    # calculate fitting line
    for condition, params in coefficients.items():
        # simulate x_values
        min_x = model_3_df[(model_3_df["subset"] == subset) & (model_3_df["condition"] == condition)]["speaker_distance"].min()
        max_x = model_3_df[(model_3_df["subset"] == subset) & (model_3_df["condition"] == condition)]["speaker_distance"].max()
        x_values = np.linspace(min_x, max_x, 100)
        
        # calculate y values
        k = params["k"]
        a = params["a"]
        y_fit = np.exp(k) * x_values**a
        
        # add data to dataframe
        fit_data.extend([(subset, condition, x, y) for x, y in zip(x_values, y_fit)])


# %% Visualisation hypothesis 3 (part I)
# get fitted dataframe and data
model_3_fit_df = pd.DataFrame(fit_data, columns=["subset", "condition", "x_values", "y_values"])

# set theme and style
analysis.define_plotting_theme()

# prepare looping
subsets = model_3_df["subset"].unique().tolist()
conditions = model_3_df["condition"].unique().tolist()

# prepare plotting
y_ticks = np.arange(2, 13, 1).tolist()
fig, axes = plt.subplots(1, 2, sharey=True)

# create every plot separately
for ax, subset in zip(axes, subsets):
    
    # plot average data with with std
    sns.pointplot(data=model_3_df[model_3_df["subset"] == subset], 
                  x="speaker_distance", y="mean_response_distance", hue="condition",
                  errorbar="sd", ls="None", native_scale=True, ax=ax,
                  markersize=5, capsize=0.2, errwidth=1,
                  palette="colorblind")
    
    # plot fitted lines
    for condition in conditions:
        condition_df = model_3_fit_df[(model_3_fit_df["subset"] == subset) & (model_3_fit_df["condition"] == condition)]
        ax.plot(condition_df["x_values"], condition_df["y_values"])
        x_ticks = np.arange(condition_df["x_values"].min(), condition_df["x_values"].max() + 1, 1).astype(int).tolist()
        
    # plot baseline
    baseline = np.linspace(model_3_df[model_3_df["subset"] == subset]["speaker_distance"].min(), model_3_df[model_3_df["subset"] == subset]["speaker_distance"].max(), 100)
    ax.plot(baseline, baseline, ls="--", color="black", alpha=0.7)
    
    # layout
    ax.set_title(subset)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("log stimulus distance [m]")
    ax.set_ylabel("log estimated distance [m]")
    ax.set_yticks(y_ticks, labels=[str(y_tick) for y_tick in y_ticks])
    ax.set_xticks(x_ticks, labels=[str(x_tick) for x_tick in x_ticks])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    
fig.legend(handles, labels, title="Stimulus exposure", loc="center left", bbox_to_anchor=(0.98, 0.5), handletextpad=0, frameon=False, alignment="left")
    
plt.tight_layout()
analysis.save_fig("hyp-3_part-1")

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
