"""stimuli utility funcs for the stroop experiment
assume red is the "dominant color"
- which should be okay since stroop task is symmetric w.r.t to color
"""
import numpy as np

# constants
COLORS = ['red', 'green']
TASKS = ['color naming', 'word reading']
CONDITIONS = ['control', 'conflict', 'congruent']

# input check
n_tasks = len(TASKS)
n_colors = len(COLORS)
assert n_colors == 2
assert n_tasks == 2


def get_color_rep(color):
    if color == 'red':
        return [1, 0]
    elif color == 'green':
        return [0, 1]
    elif color is None:
        return [0, 0]
    else:
        raise ValueError(f'Unrecognizable color: {color}')


def get_word_rep(word):
    if word == 'red':
        return [1, 0]
    elif word == 'green':
        return [0, 1]
    elif word is None:
        return [0, 0]
    else:
        raise ValueError(f'Unrecognizable word: {word}')


def get_task_rep(task):
    assert task in TASKS
    if task == 'color naming':
        return [1, 0]
    else:
        return [0, 1]


def compute_delays(SOA):
    """ calculate the delay time for color/word input
    positive SOA => color is presented earlier, v.v.

    Parameters
    ----------
    SOA : int
        stimulus onset asynchrony == color onset - word onset

    Returns
    -------
    int,int
        the delay time for color/word input, repsectively

    """
    color_delay = max(0, -SOA)
    word_delay = max(0, SOA)
    return color_delay, word_delay


def get_stimulus(
    color_input_layer, color,
    word_input_layer, word,
    task_input_layer, task,
    n_time_steps, SOA=0,
):
    """get a stroop stimulus

    Parameters
    ----------
    color/word/task_input_layer: pnl.TransferMechanism
        the input layer PNL object
    color/word/task : str
        an element in COLORS/COLORS/TASKS
    n_time_steps: int
        the stimuli sequence length
    SOA: int
        stimulus onset asynchrony; see compute_delays()

    Returns
    -------
    dict, as requested by PNL composition
        a representation of the input sitmuli sequence

    """
    assert abs(SOA) <= n_time_steps
    # set up the stimuli
    color_stimulus = np.tile(get_color_rep(color), (n_time_steps, 1))
    word_stimulus = np.tile(get_word_rep(word), (n_time_steps, 1))
    task_stimulus = np.tile(get_task_rep(task), (n_time_steps, 1))
    # onset delay
    if SOA != 0:
        color_delay, word_delay = compute_delays(SOA)
        color_stimulus[:color_delay, :] = 0
        word_stimulus[:word_delay, :] = 0
        # task_stimulus[:abs(SOA), :] = 0
    # form the input dict
    input_dict = {
        color_input_layer: color_stimulus,
        word_input_layer: word_stimulus,
        task_input_layer: task_stimulus
    }
    return input_dict


def get_stimulus_set(inp_color, inp_word, inp_task, n_time_steps, SOA=0):
    """get stimuli for all task x condition combination with some SOA

    Parameters
    ----------
    color/word/task_input_layer: pnl.TransferMechanism
        the input layer PNL object
    n_time_steps: int
        the stimuli sequence length
    SOA: int
        stimulus onset asynchrony; see compute_delays()

    Returns
    -------
    hierarchical dict
    - level 1:  key: tasks        val: stimuli for all conditions
    - level 2:  key: condition    val: a stimulus
    """
    # color naming - congruent
    inputs_cn_con = get_stimulus(
        inp_color, 'red', inp_word, 'red', inp_task, 'color naming',
        n_time_steps, SOA
    )
    # color naming - incongruent
    inputs_cn_cfl = get_stimulus(
        inp_color, 'red', inp_word, 'green', inp_task, 'color naming',
        n_time_steps, SOA
    )
    # color naming - control
    inputs_cn_ctr = get_stimulus(
        inp_color, 'red', inp_word, None, inp_task, 'color naming',
        n_time_steps, SOA
    )
    # word reading - congruent
    inputs_wr_con = get_stimulus(
        inp_color, 'red', inp_word, 'red', inp_task, 'word reading',
        n_time_steps, SOA
    )
    # word reading - incongruent
    inputs_wr_cfl = get_stimulus(
        inp_color, 'green', inp_word, 'red', inp_task, 'word reading',
        n_time_steps, SOA
    )
    # word reading - control
    inputs_wr_ctr = get_stimulus(
        inp_color, None, inp_word, 'red', inp_task, 'word reading',
        n_time_steps, SOA
    )
    # combine the stimuli to lists
    color_naming_input_list = [inputs_cn_ctr, inputs_cn_cfl, inputs_cn_con]
    word_reading_input_list = [inputs_wr_ctr, inputs_wr_cfl, inputs_wr_con]
    # for each task, pack all conditions to dictionaries
    color_naming_input_dict = dict(zip(CONDITIONS, color_naming_input_list))
    word_reading_input_dict = dict(zip(CONDITIONS, word_reading_input_list))
    # pack both tasks to a dict
    all_input_dict = dict(
        zip(TASKS, [color_naming_input_dict, word_reading_input_dict])
    )
    return all_input_dict
