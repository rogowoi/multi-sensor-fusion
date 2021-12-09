import pickle

import matplotlib.pyplot as plt
import numpy as np
from seetree.sifc_essentials import CameraTrajectory, Trajectory


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def vis_3d(graphs: list):
    """
    plot 3d graph of numpy (n, 3), where 3 is [x, y, z]
    :param graphs:
    :return:
    """
    # filtered = np.array(filtered)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    for item in graphs:
        ax.scatter(item[:, 0], item[:, 1], item[:, 2])


def get_timestamps(trajectory: Trajectory):
    timestamps = []
    for item in trajectory.collection:
        timestamps.append(item.timestamp)
    return timestamps


def np_from_traj(trajectory: Trajectory):
    positions = []
    rotations = []
    for item in trajectory.collection:
        positions.append(item.T)
        rotations.append(item.Re)
    return np.array(positions), np.array(rotations)


def np_from_traj_camera(trajectory: CameraTrajectory):
    positions = []
    rotations = []
    accs = []
    for item in trajectory.collection:
        positions.append(item.T)
        rotations.append(item.Re)
        accs.append(item.imu_acc)
    return np.array(positions), np.array(rotations), np.array(accs)


def vis_all_coords(data, timestamps, names):
    # data = [gt, filtered, original]
    num_seqs = len(data)
    num_f = data[0].shape[1]
    fig, ax = plt.subplots(1, num_f, figsize=(15, 5))
    for i in range(num_f):
        ax[i].set_ylabel(names[i])
        for j in range(num_seqs):
            item = data[j]
            ax[i].scatter(timestamps, item[:, i])