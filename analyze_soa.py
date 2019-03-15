import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stroop_stimulus_util import TASKS, CONDITIONS

"""helper funcs
"""


def compute_rt(act_r, act_g, threshold):
    """compute reaction time
    take the activity of the decision layer...
    check the earliest time point when activity > threshold...
    call that RT
    *RT=np.nan if timeout
    """
    rt_red = compute_rt_(act_r, threshold)
    rt_grn = compute_rt_(act_g, threshold)
    rt = np.nanmin([rt_red, rt_grn])
    return rt


def compute_rt_(act, threshold):
    tps_pass_threshold = np.where(act > threshold)[0]
    if len(tps_pass_threshold) > 0:
        return tps_pass_threshold[0]
    # return np.nan
    return n_time_steps


# save the record file
log_path = 'log_temp'
img_path = 'imgs_temp'

# experiment params
SOAs = [-10, -5, 0, 5, 10]
n_time_steps = 120
n_repeats = 20

# load the data record
fname = f'record_t{n_time_steps}_n{n_repeats}.csv'
record = pd.read_csv(os.path.join(log_path, fname))

# compute rt and responses
threshold = .95
column_names = [
    'Rep-id', 'Condition',
    'Task', 'SOA', 'RT'
]
df = pd.DataFrame(columns=column_names, dtype=int)

# compute reaction time
for SOA in SOAs:
    for k, task in enumerate(TASKS):
        for j, condition in enumerate(CONDITIONS):
            # print(SOA, task, condition)
            # define the current selection operator
            sel_op = (record['SOA'] == SOA) & (record['Task'] == task) & (
                record['Condition'] == condition)
            # the record for the current SOA / task / condition
            record_stc = record[sel_op]
            # loop over all repeats
            n_repeats = int(np.max(record_stc['Rep-id']))
            for i in range(n_repeats):
                # get single trial data
                record_stci = record_stc[record_stc['Rep-id'] == i]
                # compute behavioral data
                rt_stci = compute_rt(
                    record_stci['Act-Red'], record_stci['Act-Green'],
                    threshold
                )
                # shift the RT
                # ... to the earliest time point when response is possible
                if task == 'word reading':
                    soa = -SOA
                    if soa < 0:
                        rt_stci -= abs(soa)
                elif task == 'color naming':
                    soa = SOA
                    if soa < 0:
                        rt_stci -= abs(soa)
                else:
                    raise ValueError('Unrecognizable task')

                # update the data df
                df = df.append({
                    'Rep-id': i,
                    'Condition': condition,
                    'Task': task,
                    'SOA': soa,
                    'RT': rt_stci,
                }, ignore_index=True)

# show the df
df.head()

# plot the data
sns.set(
    style='white', context='talk', palette="colorblind",
    rc={'legend.frameon': False},
)

f, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.lineplot(
    x='SOA', y='RT',
    hue='Condition', style='Task', markers=True,
    ci=99,
    data=df,
    ax=ax,
)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

title_text = """
Effects of varying SOA between word & color stimuli
"""
ax.set_title(title_text)
ax.set_ylabel('Reaction time')
ax.set_xlabel('Stimulus onset asynchrony (SOA)')
f.tight_layout()
sns.despine()

f.savefig(os.path.join(img_path, 'soa.png'))
