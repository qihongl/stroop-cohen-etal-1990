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


def get_stimulus(
    color_input_layer, color,
    word_input_layer, word,
    task_input_layer, task,
    n_time_steps
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

    Returns
    -------
    dict, as requested by PNL composition
        a representation of the input sitmuli sequence

    """
    layers = [color_input_layer, word_input_layer, task_input_layer]
    stimulus_ = [get_color_rep(color), get_word_rep(word), get_task_rep(task)]
    stimulus = [np.tile(s_, (n_time_steps, 1)) for s_ in stimulus_]
    return dict(zip(layers, stimulus))
