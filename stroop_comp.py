from itertools import product
import os
import numpy as np
import psyneulink as pnl
import matplotlib.pyplot as plt
import seaborn as sns
from stroop_stimulus_util import get_stimulus, TASKS, COLORS

sns.set(
    style='white', context='poster', palette="colorblind",
    font_scale=.8, rc={"lines.linewidth": 2},
)

# log path
impath = 'imgs_temp'
if not os.path.exists(impath):
    os.makedirs(impath)

# constants
CONDITIONS = ['control', 'conflict', 'congruent']
experiment_info = f"""
stroop experiment info
- all colors:\t {COLORS}
- all words:\t {COLORS}
- all tasks:\t {TASKS}
- all conditions:\t {CONDITIONS}
- image output path = {impath}
"""
print(experiment_info)


# calculate meta-data
n_conditions = len(CONDITIONS)
n_tasks = len(TASKS)
n_colors = len(COLORS)

# model params
n_units = 2
hidden_func = pnl.Logistic(gain=1.0, x_0=4.0)
dec_noise_std = .1
unit_noise_std = 0.001
integration_rate = 0.2


# INPUT LAYER
inp_color = pnl.TransferMechanism(
    size=n_units, function=pnl.Linear, name='COLOR INPUT'
)
inp_word = pnl.TransferMechanism(
    size=n_units, function=pnl.Linear, name='WORD INPUT'
)
# TASK LAYER
inp_task = pnl.TransferMechanism(
    size=n_units, function=pnl.Linear, name='TASK'
)
# HIDDEN LAYER
hid_color = pnl.TransferMechanism(
    size=n_units,
    function=hidden_func,
    integrator_mode=True,
    integration_rate=integration_rate,
    noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
    name='COLORS HIDDEN')

hid_word = pnl.TransferMechanism(
    size=n_units,
    function=hidden_func,
    integrator_mode=True,
    integration_rate=integration_rate,
    noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
    name='WORDS HIDDEN'
)
# OUTPUT LAYER
output = pnl.TransferMechanism(
    size=n_units,
    function=pnl.Logistic,
    integrator_mode=True,
    integration_rate=integration_rate,
    noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
    name='OUTPUT'
)
# DECISION LAYER
decision = pnl.LCAMechanism(
    size=n_units,
    noise=pnl.UniformToNormalDist(standard_deviation=dec_noise_std).function,
    name='DECISION'
)

# LOGGING
hid_color.set_log_conditions('value')
hid_word.set_log_conditions('value')
output.set_log_conditions('value')

# PROJECTIONS, weights copied from cohen et al (1990)
c_ih = pnl.MappingProjection(
    matrix=[[2.2, -2.2], [-2.2, 2.2]], name='COLOR INPUT TO HIDDEN')
w_ih = pnl.MappingProjection(
    matrix=[[2.6, -2.6], [-2.6, 2.6]], name='WORD INPUT TO HIDDEN')
c_ho = pnl.MappingProjection(
    matrix=[[1.3, -1.3], [-1.3, 1.3]], name='COLOR HIDDEN TO OUTPUT')
w_ho = pnl.MappingProjection(
    matrix=[[2.5, -2.5], [-2.5, 2.5]], name='WORD HIDDEN TO OUTPUT')
proj_tc = pnl.MappingProjection(
    matrix=[[4.0, 4.0], [0, 0]], name='COLOR NAMING')
proj_tw = pnl.MappingProjection(
    matrix=[[0, 0], [4.0, 4.0]], name='WORD READING')

# build the model
comp = pnl.Composition(name='STROOP model')
comp.add_node(inp_color)
comp.add_node(inp_word)
comp.add_node(hid_color)
comp.add_node(hid_word)
comp.add_node(inp_task)
comp.add_node(output)
comp.add_node(decision)
comp.add_linear_processing_pathway([inp_color, c_ih, hid_color])
comp.add_linear_processing_pathway([inp_word, w_ih, hid_word])
comp.add_linear_processing_pathway([hid_color, c_ho, output])
comp.add_linear_processing_pathway([hid_word, w_ho, output])
comp.add_linear_processing_pathway([inp_task, proj_tc, hid_color])
comp.add_linear_processing_pathway([inp_task, proj_tw, hid_word])
comp.add_linear_processing_pathway([output, pnl.IDENTITY_MATRIX, decision])

comp.show_graph()

"""define the inputs
i.e. all CONDITIONS x TASKS for the experiment
"""
# the length of the stimulus sequence
n_time_steps = 60

# color naming - cong
inputs_cn_con = get_stimulus(
    inp_color, 'red', inp_word, 'red', inp_task, 'color naming', n_time_steps
)
# color naming - incong
inputs_cn_cfl = get_stimulus(
    inp_color, 'red', inp_word, 'green', inp_task, 'color naming', n_time_steps
)
# color naming - control
inputs_cn_ctr = get_stimulus(
    inp_color, 'red', inp_word, None, inp_task, 'color naming', n_time_steps
)
# word reading - cong
inputs_wr_con = get_stimulus(
    inp_color, 'red', inp_word, 'red', inp_task, 'word reading', n_time_steps
)
# word reading - incong
inputs_wr_cfl = get_stimulus(
    inp_color, 'green', inp_word, 'red', inp_task, 'word reading', n_time_steps
)
# word reading - control
inputs_wr_ctr = get_stimulus(
    inp_color, None, inp_word, 'red', inp_task, 'word reading', n_time_steps
)


"""run the model
test the model on all CONDITIONS x TASKS combinations
"""


def run_model(n_stimuli, inputs, execution_id):
    acts = np.zeros((n_stimuli, n_time_steps, 2))
    for i in range(n_stimuli):
        # print(f'execution_id={execution_id}')
        comp.run(
            execution_id=execution_id,
            inputs=inputs,
            num_trials=n_time_steps,
            # termination_processing=termination_op
        )
        execution_id += 1
        # log acts
        acts[i, :, :] = np.squeeze(comp.results)
    return acts, execution_id


# run the model
execution_id = 0
n_stimuli = 50
# preallocate
cn_input_list = [inputs_cn_ctr, inputs_cn_cfl, inputs_cn_con]
wr_input_list = [inputs_wr_ctr, inputs_wr_cfl, inputs_wr_con]
A_cn = {condition: None for condition in CONDITIONS}
A_wr = {condition: None for condition in CONDITIONS}

# run all conditions, color naming
for i, condition in enumerate(CONDITIONS):
    print(f'running color naming: {condition}...')
    A_cn[condition], execution_id = run_model(
        n_stimuli, cn_input_list[i], execution_id)
# run all conditions, word reading
for i, condition in enumerate(CONDITIONS):
    print(f'running word reading: {condition}...')
    A_wr[condition], execution_id = run_model(
        n_stimuli, wr_input_list[i], execution_id)

"""
plot the reaction time for all CONDITIONS x TASKS
want to recover fig 5 from cohen et al (1990)
e.g. color naming red green is slower than word reading red green
"""


def compute_rt(act, threshold=.9):
    """compute reaction time
    take the activity of the decision layer...
    check the earliest time point when activity > threshold...
    call that RT
    *RT=np.nan if timeout
    """
    n_time_steps_, n_units_ = np.shape(act)
    rts = np.full(shape=(n_units_,), fill_value=np.nan)
    for i in range(n_units_):
        tps_pass_threshold = np.where(act[:, i] > threshold)[0]
        if len(tps_pass_threshold) > 0:
            rts[i] = tps_pass_threshold[0]
    return np.nanmin(rts)


# compute RTs for color naming and word reading
RTs_cn = {condition: None for condition in CONDITIONS}
RTs_wr = {condition: None for condition in CONDITIONS}
for i, condition in enumerate(CONDITIONS):
    RTs_cn[condition] = np.array(
        [compute_rt(A_cn[condition][i, :, :])for i in range(n_stimuli)]
    )
    RTs_wr[condition] = np.array(
        [compute_rt(A_wr[condition][i, :, :])for i in range(n_stimuli)]
    )

# org data for plotting, color naming and word reading
mean_rt_cn = [np.nanmean(RTs_cn[condition]) for condition in CONDITIONS]
mean_rt_wr = [np.nanmean(RTs_wr[condition]) for condition in CONDITIONS]
std_rt_cn = [np.nanstd(RTs_cn[condition]) for condition in CONDITIONS]
std_rt_wr = [np.nanstd(RTs_wr[condition]) for condition in CONDITIONS]
xtick_vals = range(len(CONDITIONS))

# plot RT
f, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.errorbar(
    x=xtick_vals, y=mean_rt_cn, yerr=std_rt_cn, label='color naming'
)
ax.errorbar(
    x=xtick_vals, y=mean_rt_wr, yerr=std_rt_wr, label='word reading'
)

# ax.plot(
#     xtick_vals, mean_rt_cn, label='color naming')
# ax.plot(
#     xtick_vals, mean_rt_wr, label='word reading')

# ax.set_ylim([None, n_time_steps])
ax.set_ylabel('Reaction time (n cycles)')
ax.set_xticks(xtick_vals)
ax.set_xticklabels(CONDITIONS)
ax.set_xlabel('Condition')
ax.set_title('RT under various conditions')
f.legend(frameon=False, bbox_to_anchor=(1, .9))
f.tight_layout()
sns.despine()

# save fig
fname = f'stroop_ir{integration_rate}.png'
f.savefig(os.path.join(impath, fname))

"""
RT distribution
"""

f, axes = plt.subplots(n_tasks, 1, figsize=(7, 8), sharex=True, sharey=True,)

for i, condition in enumerate(CONDITIONS):
    temp = sns.kdeplot(
        RTs_cn[condition][~np.isnan(RTs_cn[condition])],
        shade=True,
        ax=axes[0]
    )
for i, condition in enumerate(CONDITIONS):
    sns.kdeplot(
        RTs_wr[condition][~np.isnan(RTs_wr[condition])],
        shade=True,
        ax=axes[1]
    )
    axes[0].legend(CONDITIONS, frameon=False)

for i, ax in enumerate(axes):
    ax.set_ylabel('Probability, KDE')
    ax.set_title(f'RT distribution, {TASKS[i]}')
axes[1].set_xlabel('Reaction time')
sns.despine()
f.tight_layout()

# save fig
fname = f'rt_kde.png'
f.savefig(os.path.join(impath, fname))

"""
inspect the hidden layer activity time course
"""


def get_log_values(condition_indices):
    """
    get the hidden layer activity, given the execution ids
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
    return hw_acts, hc_acts


def get_cond_ids(condition_index, task_index):
    """
    get indices/execution ids given the condition & task indices
    """
    c_ = condition_index
    t_ = task_index
    condition_indices_shift = (t_ * n_conditions + c_) * n_stimuli
    condition_indices = np.arange(n_stimuli) + condition_indices_shift
    return condition_indices


# get all logged values
hid_color_logvals = hid_color.log.get_logged_entries()['value']
hid_word_logvals = hid_word.log.get_logged_entries()['value']

# prep colorblind_colors for plotting
cb_r = sns.color_palette('colorblind', n_colors=4)[3]
cb_g = sns.color_palette('colorblind', n_colors=4)[2]
colors_plt = [cb_r, cb_g]
n_stds = 3

# for all tasks x conditions x hidden layers...
# plot the activity for red/green unit
for c_i, t_j in product(range(n_conditions), range(n_tasks)):
    # fetch the ex ids for that condition x task combo
    condition_indices = get_cond_ids(c_i, t_j)
    hw_acts, hc_acts = get_log_values(condition_indices)
    # plot
    f, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    # loop over the red/green units, color layer
    for i in range(n_units):
        mean_ = np.mean(hc_acts[:, :, i], axis=0)
        std_ = np.std(hc_acts[:, :, i], axis=0)
        axes[0].errorbar(
            x=range(n_time_steps), y=mean_, yerr=std_*n_stds,
            color=colors_plt[i]
        )
    # loop over the red/green units, word layer
    for i in range(n_units):
        mean_ = np.mean(hw_acts[:, :, i], axis=0)
        std_ = np.std(hw_acts[:, :, i], axis=0)
        axes[1].errorbar(
            x=range(n_time_steps), y=mean_, yerr=std_*n_stds,
            color=colors_plt[i]
        )
    # mark the plot
    for i, ax in enumerate(axes):
        ax.set_ylabel(f'Activity, {TASKS[i]}')
        ax.set_ylim([-.05, 1.05])
    axes[-1].set_xlabel('time ticks')
    axes[0].set_title(f'red trials, {TASKS[t_j]}, {CONDITIONS[c_i]}')

    sns.despine()
    f.legend(frameon=False, bbox_to_anchor=(1, .4))
    f.tight_layout()

    # save fig
    fname = f'h_{CONDITIONS[c_i]}_{TASKS[t_j]}.png'
    f.savefig(os.path.join(impath, fname))
