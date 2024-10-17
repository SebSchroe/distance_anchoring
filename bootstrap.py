import numpy as np
import pandas as pd
import analysis
from analysis import speaker_dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

cond_ids = [1, 2]
sub_ids_dict = {1: [1, 3, 4, 7], 2: [1, 2, 6]}
block_ids = [6]
speaker_distances = [6, 7, 8, 9, 10, 11]

def bs_conf(x, n_reps):
    bs_n = 10000 # number of bootstrap samples
    bs_array = np.zeros(bs_n) # this array holds calculated means (or medians in your case...)

    #the bootstrap loop
    for i in range(bs_n):
        ridx = np.random.randint(0, len(x), n_reps)  #vector of random indices in x (=selecting entries with repetition)
        bs_array[i] = np.mean(x[ridx])  #getting these random entries from x and calculating the mean in one go, then saving that

    b_mean = np.mean(x) # mean of the original data
    bias = np.abs(b_mean - np.mean(bs_array))  #any difference is called bias (you don’t need to use this)

    bs_SE = np.std(bs_array)  #the std of the means is the standard error of the mean of the original distribution!

    bs_array = np.sort(bs_array)  #sort the array of means

    bs_95conf = [bs_array[int(bs_n * 0.025)], bs_array[int(bs_n * 0.975)]] # 95% condicende interval
    
    # return (bs_95conf[1] - bs_95conf[0]) / 2
    return bs_array, bs_SE, bs_95conf, bias

# %% load data
df = analysis.get_concat_df(cond_ids=cond_ids, sub_ids_dict=sub_ids_dict, block_ids=block_ids)
df['speaker_distance'] = df['speaker_id'].apply(lambda x: analysis.get_speaker_distance(x, speaker_dict))

# %% boxplot of bootstrapped data for different n_reps and the two conditions
# filter for specific distance
filtered_df = df[df['speaker_distance'] == 11]

# create lists and empty dataframe
n_reps_list = [8, 10, 12, 15]
bs_results = pd.DataFrame(columns=['cond_id', 'n_reps', 'bs_values'])
ttest_results = []

# iterate through cond_ids and n_reps
for n_reps in n_reps_list:
    
    bs_arrays_cond1 = []
    bs_arrays_cond2 = []
    
    for cond_id in cond_ids:
        filtered_df_2 = filtered_df[filtered_df['cond_id'] == cond_id]
        response_array = filtered_df_2['response'].to_numpy()
        
        # bootstrapping
        bs_array = bs_conf(response_array, n_reps)[0]
        
        # create temporary df
        temp_df = pd.DataFrame({
            'cond_id': cond_id,
            'n_reps': n_reps,
            'bs_values': bs_array})
        
        # add temp dataframe to results dataframe
        bs_results = pd.concat([bs_results, temp_df], ignore_index=True)
        
        # t-test preparation
        if cond_id == 1:
            bs_arrays_cond1 = bs_array
        else:
            bs_arrays_cond2 = bs_array
            
    # conduction of upper one sided t-test between conditions and save results
    t_stat, p_value = stats.ttest_ind(bs_arrays_cond1, bs_arrays_cond2, alternative='greater')
    ttest_results.append((n_reps, t_stat, p_value))
    
sns.boxplot(data=bs_results, x='n_reps', y='bs_values', hue='cond_id', palette='tab10')  
    

# %% plot SEM of different n_reps
# filter for specific speaker_distance and create list for x
for speaker_distance in speaker_distances:
    filtered_df = df[df['speaker_distance'] == speaker_distance]
    response_array = filtered_df['response'].to_numpy()

    # initialize lists for SEM and n_reps
    SE_list = []
    n_reps_list = []

    for n_reps in range(1, 21, 1):
        bs_SE = bs_conf(response_array, n_reps)[1]
        SE_list.append(bs_SE)
        n_reps_list.append(n_reps)
    
    # data plotting    
    plt.scatter(n_reps_list, SE_list, label=f'at {speaker_distance} m')
    
# layout
plt.title(f'SER of 10000 bootstrapped n_reps in block {block_ids}')
plt.xlabel('n_reps')
plt.ylabel('standard error of the mean')
#plt.xticks(np.arange(0, 21, 2))
#plt.yticks(np.arange(0, 2.1, 0.2))

# show plot
plt.legend()
plt.tight_layout()
plt.show()




