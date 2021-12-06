import argparse
import time

import numpy as np
from seetree.sifc_essentials import Pose, Trajectory, CameraPose, CameraTrajectory

from trajectories import simulate, simulate_rotation, simulate_c, simulate_rotation_c, rotation_traj, trajectory


def generate(args):
    radius = args.radius

    all_tmpstamps = np.linspace(0, 360, 10000)
    timestamps = np.random.choice(all_tmpstamps, 3600)
    start_time = time.time()
    timestamps.sort()
    ms_trajectory = create_ms_trajectory(start_time, radius, timestamps, args.ms_pos_std, args.ms_rot_std)

    ms_trajectory.save_pkl(args.out_dir, 'ms_trajectory')

    zed_trajectory, zed_acc = create_zed_trajectory(start_time, radius, timestamps, args.zed_pos_std, args.zed_rot_std)

    zed_trajectory.save_pkl(args.out_dir, 'zed_trajectory')

    gt_traj = create_gt_trajectory(start_time, radius, timestamps)
    gt_traj.save_pkl(args.out_dir, 'gt_trajectory')


def create_gt_trajectory(start_time, radius, timestamps):
    rotations = []
    positions = []
    for angle in timestamps:
        pos = trajectory(angle, radius)
        rot = rotation_traj(angle)
        rotations.append(rot)
        positions.append(pos)
    positions = np.array(positions)
    rotations = np.array(rotations)
    traj = Trajectory()
    # start_time = time.time()
    i = 0
    for pos, rot in zip(positions, rotations):
        pose = Pose(i, start_time + 100 * i, T=pos, Re=rot)
        traj.add(pose)
    return traj


def create_ms_trajectory(start_time, radius, timestamps, pos_std, rot_std):
    rotations = []
    positions = []
    for angle in timestamps:
        pos = simulate(angle, radius, pos_std)
        rot = simulate_rotation(angle, radius, rot_std)
        rotations.append(rot)
        positions.append(pos)
    positions = np.array(positions)
    rotations = np.array(rotations)
    traj = Trajectory()
    # start_time = time.time()
    i = 0
    for pos, rot in zip(positions, rotations):
        pose = Pose(i, start_time + 100 * i, T=pos, Re=rot)
        traj.add(pose)
    return traj


def create_zed_trajectory(start_time, radius, timestamps, pos_std, rot_std):
    rotations = []
    positions = []
    for angle in timestamps:
        pos = simulate_c(angle, radius, pos_std)
        rot = simulate_rotation_c(angle, rot_std)
        rotations.append(rot)
        positions.append(pos)
    positions = np.array(positions)
    rotations = np.array(rotations)
    v_x = np.gradient(positions[:, 0], 1)
    v_y = np.gradient(positions[:, 1], 1)
    v_z = np.gradient(positions[:, 2], 1)

    a_x = np.gradient(v_x, 1)
    a_y = np.gradient(v_y, 1)
    a_z = np.gradient(v_z, 1)
    acc_data = np.array([a_x, a_y, a_z])

    traj = CameraTrajectory()
    # start_time = time.time()
    i = 0
    for pos, rot in zip(positions, rotations):
        pose = CameraPose(i, start_time + 1e9 * i, T=pos, Re=rot, imu_acc=acc_data)
        traj.add(pose)

    return traj, acc_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate magnetic sensor data")
    parser.add_argument('--ms_pos_std', default=0.001, type=float, required=False, help='Standard deviation of pose estimation on magnetic sensor, meters')
    parser.add_argument('--ms_rot_std', default=0.5, type=float, required=False, help='Standard deviation of rotation of magnetic sensor, degrees')
    parser.add_argument('--zed_pos_std', default=0.1, type=float, required=False,
                        help='Standard deviation of rotation of magnetic sensor, degrees')
    parser.add_argument('--zed_rot_std', default=0.5, type=float, required=False,
                        help='Standard deviation of rotation of magnetic sensor, degrees')

    parser.add_argument('--out_dir', required=True, type=str, help='Output directory to store files')
    parser.add_argument('--radius', required=False, type=int, default=2, help='Output directory to store files')
    _args = parser.parse_args()
    generate(_args)