import numpy as np


def rmse(original_seq, filtered_seq):
    diff = (original_seq - filtered_seq) ** 2
    diff_sum = np.mean(np.sqrt(diff), axis=0)
    return diff_sum, np.mean(diff_sum)