import os
import numpy as np
import pandas as pd
from stroop_model import get_stroop_model
from stroop_stimulus import get_stimulus_set
from stroop_stimulus import TASKS, COLORS, CONDITIONS

# log path
log_path = 'log_temp'
if not os.path.exists(log_path):
    os.makedirs(log_path)

# constants
experiment_info = f"""
stroop experiment info
- all colors:\t {COLORS}
- all words:\t {COLORS}
- all tasks:\t {TASKS}
- all conditions:\t {CONDITIONS}
- log path = {log_path}
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

"""run the model
test the model on all CONDITIONS x TASKS combinations
"""
execution_id = 0


def run_model(inputs, n_repeats, execution_id, record):
    rep_id = 0
    for i in range(n_repeats):
        # print(f'execution_id={execution_id}')
        model.run(
            execution_id=execution_id,
            inputs=inputs,
            num_trials=n_time_steps,
            # termination_processing=termination_op
        )
        # log acts
        acts = np.squeeze(model.results)
        # organize the data
        rep_id_vec = np.tile([rep_id], (n_time_steps, 1))
        cond_vec = np.tile([condition], (n_time_steps, 1))
        task_vec = np.tile([task], (n_time_steps, 1))
        soa_vec = np.tile([SOA], (n_time_steps, 1))
        time_vec = np.arange(0, n_time_steps).reshape(-1, 1)
        # add to the exisiting data frame
        data_stack = np.hstack(
            [rep_id_vec, cond_vec, task_vec, soa_vec, time_vec, acts]
        )
        record = record.append(
            pd.DataFrame(data_stack, columns=column_names), sort=False
        )
        # incremenet the counter
        execution_id += 1
        rep_id += 1
    return record, execution_id


# run the model with certain SOA
SOAs = [-10, 0, 10]
n_time_steps = 120
n_repeats = 20

# preallocate: data record
column_names = [
    'Rep-id', 'Condition', 'Task', 'SOA', 'Time', 'Act-Red', 'Act-Green'
]
record = pd.DataFrame(columns=column_names)

for SOA in SOAs:
    # get stimuli
    input_dict = get_stimulus_set(
        inp_color, inp_word, inp_task, n_time_steps, SOA
    )
    # loop over all tasks x conditions
    for k, task in enumerate(TASKS):
        for j, condition in enumerate(CONDITIONS):
            print(f'Running {task} - {condition}, SOA = {SOA}...')
            record, execution_id = run_model(
                input_dict[task][condition], n_repeats, execution_id, record
            )
# save the record file
fname = f'record_t{n_time_steps}_n{n_repeats}.csv'
record.to_csv(os.path.join(log_path, fname), index=False)
