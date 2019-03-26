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


"""define the inputs
i.e. all CONDITIONS x TASKS for the experiment
"""
# the length of the stimulus sequence
n_time_steps = 100

input_dict = get_stimulus_set(inp_color, inp_word, inp_task, n_time_steps)

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


"""run the model
test the model on all CONDITIONS x TASKS combinations
"""

execution_id = 0

for task in TASKS:
    for cond in CONDITIONS:
        print(f'Running {task} - {cond} ... ')
        model.run(
            execution_id=execution_id,
            inputs=input_dict[task][cond],
            num_trials=n_time_steps,
        )
        execution_id += 1


"""
data analysis
"""


def get_log_values(condition_indices):
    """
    get logged activity, given the list of execution ids
    """
    # word hidden layer
    hw_acts = np.array([
        np.squeeze(hid_word.log.nparray_dictionary()[ei]['value'])
        for ei in condition_indices
    ])
    # color hidden layer
    hc_acts = np.array([
        np.squeeze(hid_color.log.nparray_dictionary()[ei]['value'])
        for ei in condition_indices
    ])
    out_acts = np.array([
        np.squeeze(hid_color.log.nparray_dictionary()[ei]['value'])
        for ei in condition_indices
    ])
    dec_acts = np.array([
        np.squeeze(model.parameters.results.get(ei))
        for ei in condition_indices
    ])
    return hw_acts, hc_acts, out_acts, dec_acts


"""plot
"""
# collect the activity
condition_indices = range(execution_id)
hw_acts, hc_acts, out_acts, dec_acts = get_log_values(condition_indices)


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
               lw=lw_plt, label=task)
    )

"""plot the activity
"""

f, axes = plt.subplots(2, 1, figsize=(8, 8))
for j, task in enumerate(TASKS):
    for i, cond in enumerate(CONDITIONS):

        axes[0].plot(
            dec_acts[i + j*n_conditions][:, 0],
            color=col_pal[i], label=CONDITIONS[i], linestyle=lsty_plt[j],
        )
        axes[1].plot(
            dec_acts[i + j*n_conditions][:, 1],
            color=col_pal[i], linestyle=lsty_plt[j],
        )
title_text = """
Decision activity, color naming, red trial
"""
axes[0].set_title(title_text)
for i, ax in enumerate(axes):
    ax.set_ylabel(f'Activity, {COLORS[i]} unit')
axes[-1].set_xlabel('Time')

# Create the figure
axes[0].legend(handles=lgd_elements, frameon=False, bbox_to_anchor=(.85, .8))

f.tight_layout()
sns.despine()

imgname = 'dec_act.png'
f.savefig(os.path.join(img_path, imgname), bbox_inches='tight')

"""
plot dec energy
"""
data_plt = dec_acts
f, ax = plt.subplots(1, 1, figsize=(8, 4))
col_pal = sns.color_palette('colorblind', n_colors=3)
for i in np.arange(0, 3, 1):
    ax.plot(
        np.prod(np.squeeze(data_plt[i][:, :]), axis=1),
        color=col_pal[i], label=CONDITIONS[i]
    )
for i in np.arange(3, 6, 1):
    ax.plot(
        np.prod(np.squeeze(data_plt[i][:, :]), axis=1),
        color=col_pal[i-3], linestyle='--'
    )
ax.set_title(f'Decision energy over time')
ax.set_ylabel('Energy')
ax.set_xlabel('Time')


# Create the figure
ax.legend(handles=lgd_elements, frameon=False, bbox_to_anchor=(.85, .95))

f.tight_layout()
sns.despine()

imgname = 'dec_act.png'
f.savefig(os.path.join(img_path, imgname), bbox_inches='tight')
