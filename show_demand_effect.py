"""
evc scratch
"""
import os
import numpy as np
from stroop_model import get_stroop_model, N_UNITS
from stroop_stimulus import get_stimulus_set
from stroop_stimulus import TASKS, COLORS, CONDITIONS
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set(style='white', context='talk', palette="colorblind")
np.random.seed(0)

# log path
img_path = 'imgs_temp'
if not os.path.exists(img_path):
    os.makedirs(img_path)

# constants
experiment_info = f"""
stroop experiment info
- all colors:\t {COLORS}
- all words:\t {COLORS}
- all tasks:\t {TASKS}
- all conditions:\t {CONDITIONS}
- img path = {img_path}
"""
print(experiment_info)

# calculate experiment metadata
n_conditions = len(CONDITIONS)
n_tasks = len(TASKS)
n_colors = len(COLORS)

"""
get the stroop model and the stimuli
"""
model, nodes, model_params = get_stroop_model()
[integration_rate, dec_noise_std, unit_noise_std] = model_params
[inp_color, inp_word, inp_task, hid_color, hid_word, output, decision] = nodes


"""helper func
"""


def run_model(n_repeats, inputs, execution_id):
    acts = np.zeros((n_repeats, n_time_steps, N_UNITS))
    for i in range(n_repeats):
        # print(f'execution_id={execution_id}')
        model.run(
            execution_id=execution_id,
            inputs=inputs,
            num_trials=n_time_steps,
        )
        execution_id += 1
        # log acts
        acts[i, :, :] = np.squeeze(model.results)
    return acts, execution_id


"""define the inputs
i.e. all CONDITIONS x TASKS for the experiment
"""
# the length of the stimulus sequence
n_time_steps = 150
demand_levels = np.linspace(0, 1, 6)
n_demand_levels = len(demand_levels)
input_dicts = [
    get_stimulus_set(inp_color, inp_word, inp_task,
                     n_time_steps, SOA=0, demand=d)
    for d in demand_levels
]

"""run the model
test the model on all CONDITIONS x TASKS combinations
"""

execution_id = 0
for did, demand in enumerate(demand_levels):
    for task in TASKS:
        for cond in CONDITIONS:
            print(f'With demand = {demand}: running {task} - {cond} ... ')
            model.run(
                execution_id=execution_id,
                inputs=input_dicts[did][task][cond],
                num_trials=n_time_steps,
            )
            execution_id += 1


"""
data analysis
"""


def compute_rt(act, threshold=.9):
    """compute reaction time
    take the activity of the decision layer...
    check the earliest time point when activity > threshold...
    call that RT
    *RT=np.nan if timeout
    """
    n_time_steps_, N_UNITS_ = np.shape(act)
    # max_rt = n_time_steps_
    # rts = np.full(shape=(N_UNITS_,), fill_value=max_rt)
    tps_pass_threshold = np.where(act[:, 0] > threshold)[0]
    if len(tps_pass_threshold) > 0:
        return tps_pass_threshold[0]
    return n_time_steps_


def get_log_values(condition_indices):
    """
    get logged activity, given the list of execution ids
    """
    dec_acts = np.array([
        np.squeeze(model.parameters.results.get(ei))
        for ei in condition_indices
    ])
    return dec_acts


"""plot
"""
# collect the activity
condition_indices = range(execution_id)
dec_acts = get_log_values(condition_indices)


"""
setup the legend
"""
col_pal = sns.color_palette('colorblind', n_colors=3)
lsty_plt = ['-', '--']
lgd_elements = []
lw_plt = 3
for i, cond in enumerate(CONDITIONS):
    lgd_elements.append(
        Line2D([0], [0], color=col_pal[i], lw=lw_plt, label=cond))
for i, task in enumerate(TASKS):
    lgd_elements.append(
        Line2D([0], [0], linestyle=lsty_plt[i], color='black',
               lw=lw_plt, label=task))


eid = 0
threshold = .9
n_trials_total = execution_id
rt = [compute_rt(dec_acts[eid], threshold=threshold)
      for eid in range(n_trials_total)]

# plt.plot(dec_acts[1+5])
rts = np.zeros((n_demand_levels, n_tasks, n_conditions))
counter = 0
for did, demand in enumerate(demand_levels):
    for tid, task in enumerate(TASKS):
        for cid, cond in enumerate(CONDITIONS):
            rts[did, tid, cid] = compute_rt(
                dec_acts[counter], threshold=threshold
            )
            counter += 1

xticklabels = ['%.1f' % (d) for d in demand_levels]

f, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(np.mean(rts[:, 0, :], axis=1), color='black', linestyle='-')
ax.plot(np.mean(rts[:, 1, :], axis=1), color='black', linestyle='--')
ax.set_xticks(range(n_demand_levels))
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Demand')
ax.set_ylabel('Reaction time (RT)')
ax.set_title('RT as a function of the level of task demand')
f.legend(TASKS, frameon=False, bbox_to_anchor=(.9, .85))
f.tight_layout()
sns.despine()

ax.set_ylim(ylims)


f, ax = plt.subplots(1, 1, figsize=(6, 5))
clf_id = 1
n_skips = 2
ax.plot(np.arange(n_skips, n_demand_levels, 1),
        rts[n_skips:, 0, clf_id], color=col_pal[clf_id],
        label='conflicting word')
ax.plot(rts[:, 1, clf_id], color=col_pal[clf_id],
        linestyle='--', label='conflicting color')
ax.plot(rts[:, 1, 0], color=col_pal[0], linestyle='--', label='control')
ax.set_xticks(range(n_demand_levels))
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Demand')
ax.set_ylabel('Reaction time (RT)')
ax.set_title('Compared the two conflict conditions')
f.legend(frameon=False, bbox_to_anchor=(1, .85))
f.tight_layout()
sns.despine()
ylims = ax.get_ylim()
