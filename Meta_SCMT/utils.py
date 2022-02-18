import numpy as np

def h2index(h, dh):
    if isinstance(h, np.ndarray):
        h = (np.round(h/dh)).astype(int)
    else:
        h = int(round(h/dh))
    return h

def gen_decay_rate(total_steps, decay_steps):
    expected_lr_decay = 0.1
    decay_rate = np.exp(np.log(expected_lr_decay)/(total_steps/decay_steps))
    print('decay_rate: ', decay_rate)
    return decay_rate