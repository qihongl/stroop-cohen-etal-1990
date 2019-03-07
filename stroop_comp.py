from itertools import product
import os
import numpy as np
import psyneulink as pnl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(
    style='white', context='poster', palette="colorblind",
    font_scale=.8, rc={"lines.linewidth": 2},
)

# log path
impath = '../imgs'

# constants
conditions = ['control', 'conflict', 'congruent']
tasks = ['color naming', 'word reading']
colors = ['red', 'green']
n_conditions = len(conditions)
n_tasks = len(tasks)
n_colors = len(colors)

# model params
n_units = 2
hidden_func = pnl.Logistic(gain=1.0, x_0=4.0)
unit_noise_std = 0.001
integration_rate = 0.2
dec_noise_std = .1

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
#  HIDDEN LAYER
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

#   LOGGING
hid_color.set_log_conditions('value')
hid_word.set_log_conditions('value')
output.set_log_conditions('value')

#  PROJECTIONS
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

# create a composition
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

"""run the model
"""


def run_model(n_stimuli, inputs, execution_id):

    acts = np.zeros((n_stimuli, n_time_steps, 2))
    for i in range(n_stimuli):
        print(f'execution_id={execution_id}')
        execution_id += 1
        # run the model
        comp.run(
            execution_id=execution_id,
            inputs=inputs,
            num_trials=n_time_steps,
            # termination_processing=termination_op
        )
        # log acts
        acts[i, :, :] = np.squeeze(comp.results)
    return acts, execution_id


def compute_rt(act, threshold=.8):
    n_time_steps_, n_units_ = np.shape(act)
    rts = np.full(shape=(n_units_,), fill_value=np.nan)
    for i in range(n_units_):
        tps_pass_threshold = np.where(act[:, i] > threshold)[0]
        if len(tps_pass_threshold) > 0:
            rts[i] = tps_pass_threshold[0]
    return np.nanmin(rts)


# data params
n_time_steps = 50

# color naming - cong
inputs_cn_exp = {
    inp_color: np.tile([1, 0], (n_time_steps, 1)),
    inp_word: np.tile([1, 0], (n_time_steps, 1)),
    inp_task: np.tile([1, 0], (n_time_steps, 1))
}
# color naming - incong
inputs_cn_cfl = {
    inp_color: np.tile([1, 0], (n_time_steps, 1)),
    inp_word: np.tile([0, 1], (n_time_steps, 1)),
    inp_task: np.tile([1, 0], (n_time_steps, 1))
}
# color naming - control
inputs_cn_ctr = {
    inp_color: np.tile([1, 0], (n_time_steps, 1)),
    inp_word: np.tile([0, 0], (n_time_steps, 1)),
    inp_task: np.tile([1, 0], (n_time_steps, 1))
}
# word reading - cong
inputs_wr_exp = {
    inp_color: np.tile([1, 0], (n_time_steps, 1)),
    inp_word: np.tile([1, 0], (n_time_steps, 1)),
    inp_task: np.tile([0, 1], (n_time_steps, 1))
}
# word reading - incong
inputs_wr_cfl = {
    inp_color: np.tile([0, 1], (n_time_steps, 1)),
    inp_word: np.tile([1, 0], (n_time_steps, 1)),
    inp_task: np.tile([0, 1], (n_time_steps, 1))
}
# word reading - control
inputs_wr_ctr = {
    inp_color: np.tile([0, 0], (n_time_steps, 1)),
    inp_word: np.tile([1, 0], (n_time_steps, 1)),
    inp_task: np.tile([0, 1], (n_time_steps, 1))
}

# run the model
execution_id = 0
n_stimuli = 50

cn_input_list = [inputs_cn_ctr, inputs_cn_cfl, inputs_cn_exp]
wr_input_list = [inputs_wr_ctr, inputs_wr_cfl, inputs_wr_exp]

A_cn = {condition: None for condition in conditions}
A_wr = {condition: None for condition in conditions}

for i, condition in enumerate(conditions):
    print('Color naming: ', condition)
    A_cn[condition], execution_id = run_model(
        n_stimuli, cn_input_list[i], execution_id)

for i, condition in enumerate(conditions):
    print('Word reading: ', condition)
    A_wr[condition], execution_id = run_model(
        n_stimuli, wr_input_list[i], execution_id)

"""
plot the reaction time for all conditions x tasks
"""

# compute RTs
RTs_cn = {condition: None for condition in conditions}
RTs_wr = {condition: None for condition in conditions}
for i, condition in enumerate(conditions):
    RTs_cn[condition] = np.array(
        [compute_rt(A_cn[condition][i, :, :])for i in range(n_stimuli)]
    )
    RTs_wr[condition] = np.array(
        [compute_rt(A_wr[condition][i, :, :])for i in range(n_stimuli)]
    )

# org data for plotting
mean_rt_cn = [np.nanmean(RTs_cn[condition]) for condition in conditions]
mean_rt_wr = [np.nanmean(RTs_wr[condition]) for condition in conditions]
std_rt_cn = [np.nanstd(RTs_cn[condition]) for condition in conditions]
std_rt_wr = [np.nanstd(RTs_wr[condition]) for condition in conditions]
xtick_vals = range(len(conditions))

# plot RT
f, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.errorbar(
    x=xtick_vals, y=mean_rt_cn, yerr=std_rt_cn, label='color naming')
ax.errorbar(
    x=xtick_vals, y=mean_rt_wr, yerr=std_rt_wr, label='word reading')

# ax.plot(
#     xtick_vals, mean_rt_cn, label='color naming')
# ax.plot(
#     xtick_vals, mean_rt_wr, label='word reading')

# ax.set_ylim([None, n_time_steps])
ax.set_ylabel('Reaction time (n cycles)')
ax.set_xticks(xtick_vals)
ax.set_xticklabels(conditions)
ax.set_xlabel('Condition')
ax.set_title('RT under various conditions')
f.legend(frameon=False, bbox_to_anchor=(1, .9))
f.tight_layout()
sns.despine()

# save fig
fname = f'stroop_ir{integration_rate}.png'
f.savefig(os.path.join(impath, fname))

"""
inspect hidden activity time course
"""


def get_log_values(condition_indices):
    """get the hidden layer activity, given the execution ids"""
    hw_acts = np.array([
        np.squeeze(hid_word.log.nparray_dictionary()[ei]['value'])
        for ei in condition_indices
    ])
    hc_acts = np.array([
        np.squeeze(hid_color.log.nparray_dictionary()[ei]['value'])
        for ei in condition_indices
    ])
    return hw_acts, hc_acts


def get_cond_ids(condition_index, task_index):
    """get indices/execution ids given the condition & task indices"""
    c_ = condition_index
    t_ = task_index
    condition_indices_shift = (t_ * n_conditions + c_) * n_stimuli
    condition_indices = np.arange(1, n_stimuli, 1) + condition_indices_shift
    return condition_indices


# get all logged values
hid_color_logvals = hid_color.log.get_logged_entries()['value']
hid_word_logvals = hid_word.log.get_logged_entries()['value']

# prep colors
cb_r = sns.color_palette('colorblind', n_colors=4)[3]
cb_g = sns.color_palette('colorblind', n_colors=4)[2]
colors_plt = [cb_r, cb_g]

# # i is indexing the condition
# c_i = 1
# # j is indexing the task
# c_j = 0

for c_i, c_j in product(range(n_conditions), range(n_tasks)):
    # fetch the ex ids for that condition x task combo
    condition_indices = get_cond_ids(c_i, c_j)
    hw_acts, hc_acts = get_log_values(condition_indices)

    # plot
    f, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for i in range(n_units):
        mean_ = np.mean(hc_acts[:, :, i], axis=0)
        std_ = np.std(hc_acts[:, :, i], axis=0)
        # axes[0].plot(mean_)
        axes[0].errorbar(
            x=range(n_time_steps), y=mean_, yerr=std_, color=colors_plt[i])
    for i in range(n_units):
        mean_ = np.mean(hw_acts[:, :, i], axis=0)
        std_ = np.std(hw_acts[:, :, i], axis=0)
        # axes[1].plot(mean_)
        axes[1].errorbar(
            x=range(n_time_steps), y=mean_, yerr=std_, color=colors_plt[i])

    for i, ax in enumerate(axes):
        ax.set_ylabel(f'Activity, {tasks[i]}')
        ax.set_ylim([-.05, 1.05])
    axes[-1].set_xlabel('time ticks')
    axes[0].set_title(f'red trials, {tasks[c_j]}, {conditions[c_i]}')

    sns.despine()
    f.legend(frameon=False, bbox_to_anchor=(1, .4))
    f.tight_layout()

    # save fig
    fname = f'h_{conditions[c_i]}_{tasks[c_j]}.png'
    f.savefig(os.path.join(impath, fname))
