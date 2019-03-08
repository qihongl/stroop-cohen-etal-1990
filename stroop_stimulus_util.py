import numpy as np

# constants
TASKS = ['color naming', 'word reading']
COLORS = ['red', 'green']

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
    positive SOA => word is presented earlier, v.v.
    for example, if...
    - color presented starting from 0ms
    - word  presented starting from 100ms
    - SOA == -100 < 0 => color presented earlier <=> delay word presentation

    Parameters
    ----------
    SOA : int
        stimulus onset asynchrony == color onset - word onset

    Returns
    -------
    int,int
        the delay time for color/word input, repsectively

    """
    color_delay = max(0, abs(SOA))
    word_delay = abs(min(0, SOA))
    return color_delay, word_delay


def get_stimulus(
    color_input_layer, color,
    word_input_layer, word,
    task_input_layer, task,
    n_time_steps, SOA=0
):
    """get a stroop stimulus

    Parameters
    ----------
    color/word/task_input_layer : pnl.TransferMechanism
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
    # form the input dict
    input_dict = {
        color_input_layer: color_stimulus,
        word_input_layer: word_stimulus,
        task_input_layer: task_stimulus
    }
    return input_dict


# def get_stimulus_(
#     color_input_layer, color,
#     word_input_layer, word,
#     task_input_layer, task,
#     n_time_steps
# ):
#     """get a stroop stimulus
#
#     Parameters
#     ----------
#     color/word/task_input_layer : pnl.TransferMechanism
#         the input layer PNL object
#     color/word/task : str
#         an element in COLORS/COLORS/TASKS
#     n_time_steps: int
#         the stimuli sequence length
#
#     Returns
#     -------
#     dict, as requested by PNL composition
#         a representation of the input sitmuli sequence
#
#     """
#     layers = [color_input_layer, word_input_layer, task_input_layer]
#     stimulus_ = [get_color_rep(color), get_word_rep(word), get_task_rep(task)]
#     stimulus = [np.tile(s_, (n_time_steps, 1)) for s_ in stimulus_]
#     return dict(zip(layers, stimulus))
