import pathlib
import os
import LinearRegDiagnostic
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.power as smp
from scipy.stats import shapiro, kstest, ttest_ind

DIR = pathlib.Path(os.curdir)
speaker_dict = {0: 2.00,
                1: 3.00,
                2: 4.00,
                3: 5.00,
                4: 6.00,
                5: 7.00,
                6: 8.00,
                7: 9.00,
                8: 10.00,
                9: 11.00,
                10: 12.00}

def plot_data_per_sub(df, sub_id, x, y, baseline=False, save=False):
    
    # filter df per sub_id
    sub_id_int = int(sub_id)
    filtered_df = df[df["sub_id"] == sub_id_int]
    
    # calculate mean and std data per speaker_distance
    means_df = (
        filtered_df
        .groupby(["block_id", x], as_index=False)
        .agg(
            mean_y_value=(y, "mean"),
            std_y_value=(y, "std")
            )
        )
    
    # create FacetGrid
    g = sns.FacetGrid(filtered_df, col="block_id", height=4.5)
    
    # scatterplot
    g.map_dataframe(sns.scatterplot, x=x, y=y, color="grey", alpha=0.5, label="Data points")
    
    for ax, (block_id, sub_df) in zip(g.axes.flat, means_df.groupby("block_id")):
        
        # add lineplot and errorbars
        ax.errorbar(x=sub_df[x], y=sub_df["mean_y_value"], yerr=sub_df["std_y_value"],
                    fmt="-o", markersize=5, capsize=4, label="Mean ± Std")
        
        #ax.set_aspect("equal", adjustable="box")
        
        if baseline == "one_one":
            # add 1:1 line through the origin
            ax.plot([2, 12], [2, 12], ls="--", color="grey", label="1:1 Line") # add 1:1 line through the origin
        elif baseline == "zero":
            ax.plot([0, 12], [0, 0], ls="--", color="grey", label="zero line") # add baseline at y = 0
        else:
            continue # no baseline
    
    # get labels for legend
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    
    # adjust layout    
    g.fig.legend(handles, labels, title="Legend", loc="center right")
    g.fig.suptitle(f"{y} per {x} (sub-{sub_id})")
    plt.subplots_adjust(right=0.92, top=0.85)
    
    if save:
        save_plot(sub_id=sub_id, x=x, y=y)
    else:
        plt.show()

def plot_averaged_data(df, x, y):
    
    # calculate mean response_distance, mean signed and mean absolute error
    means_df = get_means_df(df=df, value_to_mean=y)
    # calculate means of mean response_distance, mean signed and mean absolute error
    mean_of_means_df = get_mean_of_means_df(df=means_df, mean_value_to_mean=f"mean_{y}")
    
    if y == "response_distance":
        baseline = "one_one"
    elif y == "signed_error":
        baseline = "zero"
    else:
        baseline = None
    
    # plot mean results of each sub_id per cond_id and block_id
    plot_data(df=means_df, x=x, y=f"mean_{y}",
              col="block_id", row="cond_id", hue="sub_id", kind="lineplot", baseline=baseline)

    # plot mean of mean results with error bars
    plot_with_error_bars(df=mean_of_means_df, x=x, y=f"mean_mean_{y}",
                         yerr=f"std_mean_{y}", col="block_id", row="cond_id", baseline=baseline)

    # plot boxplot of mean results
    plot_boxplot(df=means_df, x=x, y=f"mean_{y}",
                 col="block_id", hue="cond_id", baseline=baseline)
    
    return means_df, mean_of_means_df


def save_plot(sub_id, x, y):
    
    # convert sub_id
    sub_id_int = int(sub_id)
    
    # set cond_id
    if sub_id_int <= 30:
        if sub_id_int % 2 == 1:
            cond_id = 1
        else:
            cond_id = 2
    else:
        cond_id = 3
        
    save_path = DIR / "analysis" / "individual_results_raw_data" / f"cond-{cond_id}" / f"{y}_per_{x}" / f"sub-{sub_id}_{y}_per_{x}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Plot for sub_{sub_id} was saved unter {save_path}")


def identify_and_remove_response_outliers(df):
    
    # prepare outlier removal
    cleaned_df = df.copy()
    removal_summary = []
    total_removals = 0
    
    for sub_id in cleaned_df["sub_id"].unique():
        
        sub_id_int = int(sub_id)
        sub_df = cleaned_df[cleaned_df["sub_id"] == sub_id_int]
    
        for block_id in sub_df["block_id"].unique():
            
            # filter for block_ids
            block_df = sub_df[sub_df["block_id"] == block_id]
            
            # calculate outlier border
            mean = block_df["response_time"].mean()
            std = block_df["response_time"].std()
            sd3 = mean + 3 * std
            
            # identify outliers
            if sub_id == 6 and block_id == 1:
                block_outliers = block_df # remove whole block_ dataset
            else:
                block_outliers = block_df[block_df["response_time"] > sd3]
            
            # drop outliers
            cleaned_df = cleaned_df.drop(index=block_outliers.index)
            
            # add counted outliers to summary
            removal_summary.append(f"Sub-{sub_id} in Block-{block_id}: 3sd = {sd3:.3f} -> {len(block_outliers)} response outliers removed.")
            total_removals += len(block_outliers)
                
    print(f"\nA total of {total_removals} response outliers have been removed:")
    print("\n".join(removal_summary))
    print("\nBlock 1 of sub-6 has been removed totally due to too long response time")
    
    return cleaned_df

def detect_and_remove_outliers_with_IQR(df, cond_id, y):
    
    print(f"\nChecking for assumption 2 in group {cond_id}.")
    
    cond_df = df[df["cond_id"] == cond_id]
    
    q1 = cond_df[f"mean_{y}"].quantile(0.25)
    q3 = cond_df[f"mean_{y}"].quantile(0.75)
    
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    cleaned_df = cond_df[(cond_df[f"mean_{y}"] >= lower_bound) &
                       (cond_df[f"mean_{y}"] <= upper_bound)]
    
    detected_outliers = len(cond_df) - len(cleaned_df)
    
    if detected_outliers == 0:
        print(f"No outliers in group {cond_id} detected. -> Assumption 2 for group {cond_id} is True")
        return cond_df
    
    print(f"Outliers for group {cond_id} detected: {detected_outliers}")
    # interactive decission if outliers will be removed
    user_input = input("Remove outliers from this dataframe? (y/n): ").strip().lower()
    if user_input == "y":
        print(f"Outliers for group {cond_id} removed. -> Assumption 2 is True")
        return cleaned_df
    elif user_input == "n":
        print(f"No outliers for group {cond_id} removed -> Assumption 2 is False")
        return cond_df        

def plot_data(df, x, y, col, row, hue, kind="scatterplot", baseline=False):
    
    # data plotting
    g = sns.FacetGrid(df, col=col, row=row, hue=hue, palette="tab10")
    
    if kind == "scatterplot":
        g.map(sns.scatterplot, x, y, alpha=0.5).add_legend()
    elif kind == "lineplot":
        g.map(sns.lineplot, x, y, marker="o", alpha=0.5).add_legend()
    elif kind == "regplot":
        g.map(sns.regplot, x, y, order=2).add_legend()
    
    g.add_legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    
    # adjust layout
    for ax in g.axes.flat:
        # ax.set_aspect("equal", adjustable="box")
        
        if baseline == "one_one":
            # add 1:1 line through the origin
            ax.plot([2, 12], [2, 12], ls="--", color="grey", label="1:1 Line") # add 1:1 line through the origin
        elif baseline == "zero":
            ax.plot([2, 12], [0, 0], ls="--", color="grey", label="zero line") # add baseline at y = 0
        else:
            continue # no baseline
        
    plt.show()

def plot_with_error_bars(df, x, y, yerr, col, row, baseline=False):
    
    # create FacetGrit
    g = sns.FacetGrid(df, col=col, row=row, palette="tab10")
    
    # map data to the grid
    g.map_dataframe(sns.lineplot, x=x, y=y, marker="o")
    
    for ax, (_, sub_df) in zip(g.axes.flat, g.facet_data()):
        # add error bars
        ax.errorbar(sub_df[x], sub_df[y], yerr=sub_df[yerr], fmt="none", color="black", capsize=3)
        
        if baseline == "one_one":
            ax.plot([2, 12], [2, 12], ls="--", color="grey", label="1:1 Line") # add 1:1 line through the origin
        elif baseline == "zero":
            ax.plot([2, 12], [0, 0], ls="--", color="grey", label="zero line") # add baseline at y = 0
        else:
            continue # no baseline
        
        # set equal aspect
        # ax.set_aspect("equal", adjustable="box")
        
    # layout
    g.tight_layout()
    plt.show()
    
def plot_boxplot(df, x, y, col, hue, baseline=False):
    
    # create FacetGrit
    g = sns.FacetGrid(df, col=col)
    
    # # map data to the grid
    g.map_dataframe(sns.boxplot, x=x, y=y, hue=hue, dodge=True, palette="tab10")

    # adjust layout
    for ax in g.axes.flat:
        
        if baseline == "one_one":
            ax.plot([0, 10], [2, 12], ls="--", color="grey", label="1:1 Line") # add 1:1 line through the origin
        elif baseline == "zero":
            ax.plot([0, 10], [0, 0], ls="--", color="grey", label="zero line") # add baseline at y = 0
        else:
            continue # no baseline

        # Set equal aspect ratio
        # ax.set_aspect("equal", adjustable="box")
    
    g.add_legend(title="Condition")
    plt.show() 

# distributions
def show_data_distribution(df, x):
    
    # create array of data and make shure that values are numeric
    array = df[x].to_numpy()
    
    # prepare multiplot
    fig,axes = plt.subplots(1, 2)
        
    # plot histogram
    sns.histplot(data=df, x=x, kde=True, ax=axes[0])
    axes[0].set_title("Histogram with KDE")
    
    # plot QQ-Plot
    sm.qqplot(array, line="s", ax=axes[1])
    axes[1].set_title("QQ-Plot")
    
    plt.tight_layout()
    plt.show()
    
    print("\nChecking for normality:")
    
    # Shapiro-Wilk Test
    test_stats, p_value = shapiro(array)
    print("Result of Shapiro-Wilk Test:")
    print(f"Statistic: {test_stats:.3f}, p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("Reject H0: Data is not Gaussian. -> Assumption 3 is False")
    else:
        print("Fail to reject H0: Data is Gaussian. -> Assumption 3 is True")
    
    # Kolmogorov-Smirnov Test
    # ks_stat, ks_p_value = kstest(array, "norm", args=(np.mean(array), np.std(array)))
    # print("\nResult of Kolmogorov-Smirnov Test:")
    # print(f"Statistic: {ks_stat:.3f}, p-value: {ks_p_value:.3f}")
    # if ks_p_value < 0.05:
    #     print("Reject H0: Data is not Gaussian.")
    # else:
    #     print("Fail to reject H0: Data is Gaussian.")
        
    

def get_group_parameter(array):
    
    mean = array.mean()
    std = array.std()
    n = len(array)
    
    parameter = [mean, std, n]
    
    return parameter

# statistical power analysis
def calculate_cohens_d(mean_1, std_1, n_1, mean_2, std_2, n_2):    
    pooled_std = np.sqrt(((n_1 - 1) * std_1 ** 2 + (n_2 - 1) * std_2 ** 2) / (n_1 + n_2 - 2))
    d = (mean_1 - mean_2) / pooled_std
    return d

def statistical_power(group_1, group_2, alpha=0.05, alternative="two-sided"):
    """
    alternative = two-sided", "larger" or "smaller"
    """
    mean_1, std_1, n_1 = group_1
    mean_2, std_2, n_2 = group_2
    
    nobs1 = n_1
    ratio = n_2 / n_1
    
    effect_size = calculate_cohens_d(mean_1, std_1, n_1, mean_2, std_2, n_2)
    
    analysis = smp.TTestIndPower()
    power = analysis.solve_power(effect_size=effect_size, alpha=alpha, nobs1=nobs1, ratio=ratio, alternative=alternative)
    
    print("\nStatistical power analysis:")
    print(f"Group 1: mean = {mean_1:.3f}, std = {std_1:.3f}, n: {n_1}")
    print(f"Group 2: mean = {mean_2:.3f}, std = {std_2:.3f}, n: {n_2}")
    print(f"Cohen's d: {effect_size:.3f}")
    print(f"-> Statistical power of given groups: {power:.3f}")

# linear regression diagnostics
def create_diagnostic_plots(df, x, y):
    """
    1. .residual_plot() -> Residuals vs Fittes values: checks if relationship between x and y is linear (linearity)
    2. .qq_plot() -> Normal Q-Q: checks if errors/residuals are normally distibuted (normality)
    3. .scale_location_plt() -> Scale-location: checks if the residual-variance is the same for every value of x (homoskedasticity)
    4. .leverage_plot() -> Residuals vs Leverage: checks if observations are independent of each other (outliers)
    """
    
    # fitting linear model
    model = smf.ols(formula=f"{y} ~ {x}", data=df).fit() # formula = y ~ x
    print(model.summary())
    
    # generate diagnostic plots
    cls = LinearRegDiagnostic.LinearRegDiagnostic(model)
    vif, fig, ax = cls()
    print(vif)

# help functions
def get_Basti_df(sub_ids, cond_ids, block_ids):
    
    # create empty dataframe
    df = pd.DataFrame()
    
    # loop through all sub_ids
    for sub_id in sub_ids:
        sub_dir = DIR / "results" / f"sub-{sub_id}"
        
        # load all containing result files
        for file_path in sub_dir.glob("*.txt"):
            try:
                # get csv file and concatenate
                new_df = pd.read_csv(file_path)
                df = pd.concat([df, new_df], axis=0, ignore_index=True)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_path}")
    
    # filter for specific cond_ids and block_ids
    df = df[df["cond_id"].isin(cond_ids) & df["block_id"].isin(block_ids)]
    print(f"\nFiltered for conditions {cond_ids} and blocks {block_ids}")
    
    # get number of sub_ids per condition
    print("\nn cond_1:", sum(int(sub_id) % 2 != 0 for sub_id in sub_ids))
    print("n cond_2:", sum(int(sub_id) % 2 == 0 for sub_id in sub_ids))
    
    print("\nData treatment for Basti_df:")
    df = df.rename(columns={"led_distance": "response_distance"})
    print("Renamed led_distance to response_distance.")
    
    df["stim_id"] = df["stim_id"].str.strip() # convert values of stim_id to same form
    print("Converted stim_id values in same form.")
    
    df["response_time"] = pd.to_numeric(df["response_time"]) # convert values in numeric values
    print("Converted response_time to numeric values.")
    
    df["speaker_distance"] = df["speaker_id"].apply(lambda x: get_speaker_distance(x, speaker_dict)) # convert speaker_id in  actual distance
    print("Speaker IDs have been converted to speaker distance.")
    
    return df

def get_questionnaire_df():
    
    file_path = DIR / "results" / "questionnaire_results.csv"
    df = pd.read_csv(file_path, sep=";")
    df["sub_id"] = df["sub_id"].str.replace("sub-", "").astype(int)
    return df

def get_Oskar_df():
    
    file_path = DIR / "results" / "distance_plasticity.csv"
    df = pd.read_csv(file_path)
    
    # data treatment
    print("\nData treatment for Oskar_df:")
    df = df.rename(columns={"subject_ID": "sub_id", 
                            "spk_dist": "speaker_distance", 
                            "channel": "speaker_id",
                            "slider_dist": "response_distance", 
                            "USO_file_name": "stim_id",
                            "idx": "event_id",
                            "phase": "block_id",
                            "signed_err": "signed_error",
                            "absolute_err": "absolute_error"})
    print("Renamend columns to better fit Basti_df")
    
    df["cond_id"] = 3 # add column with cond 3
    print("Added column with cond_id = 3")
    
    df["sub_id"] = df["sub_id"].str.replace("sub_", "").astype(int)
    df["sub_id"] = df["sub_id"] + 30
    print("Converted sub_id Strings to Integer and added 30 to every sub_id")
    
    df["block_id"] = df["block_id"].astype(int)
    # df = df[df["block_id"] != 3]
    df["block_id"] = df["block_id"].replace({0: 1, 1: 2, 2: 4, 3: 6})
    print("Mapped phases with block_ids of Basti_df: 0 -> 1, 1 -> 2, 2 -> 4, 3 -> 6")
    
    return df

def merge_dataframes(df_1, df_2):
    
    columns_to_merge = ["sub_id", "cond_id", "block_id", "event_id", "stim_id", 
                        "speaker_distance", "response_distance", "response_time"]
    
    df_1 = df_1[columns_to_merge]
    df_2 = df_2[columns_to_merge]
    
    merged_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
    
    return merged_df

def observe_questionnaire(df, x, y, hue):
    
    sns.set_palette("tab10")
    
    sns.stripplot(data=df, x=x, y=y, hue=hue, dodge=True, color="grey", alpha=0.5, legend=False)
    
    sns.violinplot(data=df, x=x, y=y, hue=hue)
    
    plt.show()

def remove_failed_responses(df):
    
    rows_pre = len(df)
    df = df[df["response_time"] > 0.5]
    rows_post = len(df)
    removed_rows = rows_pre - rows_post
    
    print(f"\nA total of {removed_rows} failed responses have been removed.")
    return df

def data_calculations(df):
    
    df["signed_error"] = df["response_distance"] - df["speaker_distance"] # calculate signed error
    df["absolute_error"] = abs(df["signed_error"]) # calculate absolute error
    df["squared_signed_error"] = df["signed_error"] ** 2 # calculate squared signed error
    print("\nSigned, absolute and squared signed error per trial have been calculated")
    
    return df

def get_means_df(df, value_to_mean):
    
    df = (
        df.groupby(["sub_id", "cond_id", "block_id", "speaker_distance"], as_index=False)
        .agg(mean_value=(f"{value_to_mean}", "mean"),
             std_value=(f"{value_to_mean}", "std")
        ))
    
    df = df.rename(columns={"mean_value": f"mean_{value_to_mean}", "std_value": f"std_{value_to_mean}"})
    return df

def get_mean_of_means_df(df, mean_value_to_mean):
    
    df = (
        df.groupby(["cond_id", "block_id", "speaker_distance"])
        .agg(mean_mean_value=(f"{mean_value_to_mean}", "mean"),
             std_mean_value=(f"{mean_value_to_mean}", "std")
             )
        .reset_index()
    )
    
    df = df.rename(columns={"mean_mean_value": f"mean_{mean_value_to_mean}", "std_mean_value": f"std_{mean_value_to_mean}"})
    return df

def calc_experiment_duration(n_reps, mean_response_time):
    n_reps = n_reps
    n_speaker = [5, 11]
    isi = [0.3, 2]

    n_trials_1 = n_reps * n_speaker[0]
    n_trials_2 = n_reps * n_speaker[1]

    time_per_trial_1 = mean_response_time + isi[0]
    time_per_trial_2 = isi[1]

    experiment_duration_m = (3 * (n_trials_1 * time_per_trial_1) + 2 * (n_trials_1 * time_per_trial_2) + (n_trials_2 * time_per_trial_1))/60
    return experiment_duration_m

def get_speaker_distance(key, speaker_dict):
    return speaker_dict.get(key, None)
