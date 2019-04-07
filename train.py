"""
evc scratch
"""
import os
import numpy as np
from collections import Counter
from itertools import product
from stroop_model import get_stroop_model_
from stroop_stimulus import get_stimulus_set_train
from stroop_stimulus import TASKS, COLORS, CONDITIONS
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
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
# turn off noise
model, nodes = get_stroop_model_()
[inp_color, inp_word, hid_color, hid_word, output] = nodes
# model.show_graph()
"""define the inputs
i.e. all CONDITIONS x TASKS for the experiment
"""
# model.show_graph()


def get_target(input_dict_, task):
    """helper func to add target to the input dict
    """
    assert task in TASKS
    if task == 'color naming':
        target = input_dict_[inp_color]
    else:
        target = input_dict_[inp_word]
    return target


# the length of the stimulus sequence
stimuli = get_stimulus_set_train(inp_color, inp_word)
input_dicts = {
    task: {color: None for color in COLORS}
    for task in TASKS
}
for task, color in product(TASKS, COLORS):
    input_dicts[task][color] = {
        'inputs': stimuli[task][color],
        'targets': {output: get_target(stimuli[task][color], task)},
        'epochs': 1
    }


"""run the model
test the model on all CONDITIONS x TASKS combinations
"""


def pick_task(prob_word_reading=.9):
    if np.random.uniform() < prob_word_reading:
        task = 'word reading'
    else:
        task = 'color naming'
    return task


n_epochs = 1000
report_freq = 100

for ep in range(n_epochs):
    color = np.random.choice(COLORS)
    task = pick_task()
    model.run(
        # execution_id=execution_id,
        inputs=input_dicts[task][color],
        num_trials=1,
    )
    if np.mod(ep, report_freq) == 0:
        print(ep)
    # execution_id += 1


f, ax = plt.subplots(1, 1, figsize=(10, 5))
def_eid = model.default_execution_id
ax.plot(model.parameters.losses.get(def_eid))
ax.set_xlabel('Epochs')
ax.set_ylabel('Average loss')
# ax.set_title('Learning curve')
sns.despine()

model.parameters
model.get_parameters()


# task_freq = Counter(task_log)
# cond_freq = Counter(cond_log)
