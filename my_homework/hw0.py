'''
hw0.py
Author: Camelia D. Brumar

Tufts COMP 135 Intro ML, Fall 2020

Summary
-------
Complete the problems below to demonstrate your mastery of numerical Python.
Submit to the autograder to be evaluated.
You can do a basic check of the doctests via:
$ python -m doctest hw0.py
'''

import numpy as np


def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    L = x_all_LF.shape[0]
    N = int(np.ceil(frac_test * L))

    permuted_x_all_LF = random_state.permutation(x_all_LF)

    # We want the test set to be a uniform at random subset.
    # shuffle
    # permutation

    M = L - N
    train_MF = permuted_x_all_LF[:M]
    test_NF = permuted_x_all_LF[M:]

    return train_MF, test_NF


def euclidian_distance(v_q, v_n):
    return np.sqrt(np.sum((v_n - v_q) ** 2))


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    neighb_QKF = []

    for i in range(0, query_QF.shape[0]):
        v_q = query_QF[i];
        dist_qn = [euclidian_distance(v_q, v_n) for v_n in data_NF]

        # idxs = np.argpartition(dist_qn, K)
        idxs = sorted(range(len(dist_qn)), key=lambda k: dist_qn[k]) #TODO can I use sorted??

        neighb_qi = np.asarray([data_NF[idx] for idx in idxs[:K]])
        neighb_QKF.append(neighb_qi)

    # print(neighb_QKF)
    return np.array(neighb_QKF)