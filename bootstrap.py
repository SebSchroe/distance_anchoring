import numpy as np
import analysis
from analysis import speaker_dict
import matplotlib.pyplot as plt

cond_ids = [1]
sub_ids_dict = {1: [7]}
block_ids = [2]

# %% das hier mache ich 1000 mal für bs_n 2, 4, 6, 8, 10, 15 und 20
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

    return bs_SE

# %% load and transform data
df = analysis.get_concat_df(cond_ids=cond_ids, sub_ids_dict=sub_ids_dict, block_ids=block_ids)
df['speaker_distance'] = df['speaker_id'].apply(lambda x: analysis.get_speaker_distance(x, speaker_dict))

# %% filter for specific speaker_distance and create list for x
filtered_df = df[df['speaker_distance'] == 12]
response_array = filtered_df['response'].to_numpy()

SE_list = []
n_reps_list = []

for n_reps in range(1, 21, 1):
    bs_SE = bs_conf(response_array, n_reps)
    SE_list.append(bs_SE)
    n_reps_list.append(n_reps)
    
plt.scatter(n_reps_list, SE_list)
plt.xlim(0, 21)
plt.ylim(0, 2)
plt.show()




