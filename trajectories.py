import numpy as np
import math


def trajectory(angle, radius):
    xs = radius * np.sin(angle * 2 * np.pi / 360)
    ys = radius * np.cos(angle * 2 * np.pi / 360)
    spherical = angle * 2 * np.pi / 360 * 2 * np.pi * radius
    zs = 4 * np.sin(spherical)  # * np.cos(spherical)
    return xs, ys, zs


def std_from_r(start_std, radius):
    return start_std * np.exp(2 * radius)


def simulate(angles, radius, base_std=0.001):
    xs, ys, zs = trajectory(angles, radius)
    std = std_from_r(base_std, radius)
    return xs + np.random.normal(0, std), ys + np.random.normal(0, std), zs + np.random.normal(0, std)


def simulate_c(angles, radius, base_std=0.001):
    xs, ys, zs = trajectory(angles, radius)
    # std = std_from_r(base_std, radius)
    return xs + np.random.normal(0, base_std), ys + np.random.normal(0, base_std), zs + np.random.normal(0, base_std)


# yaw, pitch, roll
def _calculate_R(roll, pitch, yaw):
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll * math.pi / 180), -math.sin(roll * math.pi / 180)],
        [0.0, math.sin(roll * math.pi / 180), math.cos(roll * math.pi / 180)]
    ], dtype=np.float)

    Ry = np.array([
        [math.cos(pitch * math.pi / 180), 0.0, math.sin(pitch * math.pi / 180)],
        [0.0, 1.0, 0.0],
        [-math.sin(pitch * math.pi / 180), 0.0, math.cos(pitch * math.pi / 180)]
    ], dtype=np.float)

    Rz = np.array([
        [math.cos(yaw * math.pi / 180), -math.sin(yaw * math.pi / 180), 0.0],
        [math.sin(yaw * math.pi / 180), math.cos(yaw * math.pi / 180), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float)
    return np.matmul(np.matmul(Rx, Ry), Rz)


def rotation_traj(angle):
    roll = 0
    pitch = 0
    yaw = 90 - angle
    return roll, pitch, yaw


def simulate_rotation(angle, radius, base_std=0.5):
    roll, pitch, yaw = rotation_traj(angle)
    std = std_from_r(base_std, radius)
    return roll + np.random.normal(0, std), pitch + np.random.normal(0, std), yaw + np.random.normal(0, std)


def simulate_rotation_c(angle, std=0.5):
    roll, pitch, yaw = rotation_traj(angle)
    # std = std_from_r(base_std, radius)
    return roll + np.random.normal(0, std), pitch + np.random.normal(0, std), yaw + np.random.normal(0, std)


def get_view_vector(rotation, position):
    r, p, yaw = rotation
    R = _calculate_R(r, p, yaw)
    view = np.dot(R, [-1, 0, 0])
    return position, position + view
