import pickle

import numpy as np

from KalmanFilter import KalmanFilter


def movement(t):
    return [np.sin(2 * t) * np.cos(3 * t), np.sin(t) * np.cos(2 * t), np.sin(4 * t)]

def rotation(t):
    return [np.sin(t) * np.cos(2 * t), np.sin(2 * t) * np.cos(t), np.cos(2 * t)]

def acceleration(t):
    return [-13 * np.sin(2 * t) * np.cos(3 * t) - 12 * np.sin(3 * t) * np.cos(2 * t),
            -4 * np.sin(t) * np.cos(2 * t) - 5 * np.sin(2 * t) * np.cos(t),
            -16 * np.sin(4 * t)]

def imu(r, p, y):
    return r + np.random.normal(0, 1), p + np.random.normal(0, 5), y + np.random.normal(0, 1)

def magnetic_pose(x, y, z, roll, pitch, yaw):
    return x + np.random.normal(0, 1), y + np.random.normal(0, 1), z + np.random.normal(0, 4), \
           roll + np.random.normal(0, 5), pitch + np.random.normal(0, 20), yaw + np.random.normal(0, 5)

def accelarometer(x, y, z):
    return x + np.random.normal(0, 10), y + np.random.normal(0, 10), z + np.random.normal(0, 10)


def main():
    dt = 1
    IMU_KF = KalmanFilter(dt, 0, 0, 0, 0.1, 1, 5, 1)
    acc_KF = KalmanFilter(dt, 0, 0, 0, 0.1, 10, 10, 10)
    xyz_KF = KalmanFilter(dt, 0, 0, 0, 0.1, 1, 1, 4)
    magn_KF = KalmanFilter(dt, 0, 0, 0, 0.1, 5, 20, 5)

    init_t = 0
    num_iterations = 1000
    real = []
    real_a = []
    real_m = []
    real_mr = []
    corrected = []
    corrected_a = []
    corrected_m = []
    corrected_mr = []
    initial = []
    initial_a = []
    initial_m = []
    for i in range(num_iterations):
        t = init_t + i * dt
        roll, pitch, yaw = rotation(t)
        ax, ay, az = acceleration(t)
        x, y, z = movement(t)
        initial.append((roll, pitch, yaw))
        initial_a.append((ax, ay, az))
        initial_m.append((x, y, z))

        roll_n, pitch_n, yaw_n = imu(roll, pitch, yaw)
        ax_n, ay_n, az_n = accelarometer(ax, ay, az)
        x_n, y_n, z_n, r_n, p_n, ya_n = magnetic_pose(x, y, z, roll, pitch, yaw)
        real.append((roll_n, pitch_n, yaw_n))
        real_a.append((ax_n, ay_n, az_n))
        real_m.append((x_n, y_n, z_n))
        real_mr.append((r_n, p_n, ya_n))
        IMU_KF.predict()
        acc_KF.predict()
        xyz_KF.predict()
        magn_KF.predict()
        # predicted.append((x_p, y_p, z_p))
        (roll_l, pitch_l, yaw_l) = IMU_KF.update([roll_n, pitch_n, yaw_n])
        (ax_l, ay_l, az_l) = acc_KF.update([ax_n, ay_n, az_n])
        (x_l, y_l, z_l) = xyz_KF.update([x_n, y_n, z_n])
        (r_l, p_l, ya_l) = acc_KF.update([r_n, p_n, ya_n])
        corrected.append((roll_l, pitch_l, yaw_l))
        corrected_a.append((ax_l, ay_l, az_l))
        corrected_m.append((x_l, y_l, z_l))
        corrected_mr.append((r_l, p_l, ya_l))

    with open('initialIMU.pickle', 'wb') as f:
        pickle.dump(initial, f)

    with open('realIMU.pickle', 'wb') as f:
        pickle.dump(real, f)

    with open('correctedIMU.pickle', 'wb') as f:
        pickle.dump(corrected, f)

    with open('initialAcc.pickle', 'wb') as f:
        pickle.dump(initial_a, f)

    with open('realAcc.pickle', 'wb') as f:
        pickle.dump(real_a, f)

    with open('correctedAcc.pickle', 'wb') as f:
        pickle.dump(corrected_a, f)

    with open('initialM.pickle', 'wb') as f:
        pickle.dump(initial_m, f)

    with open('realM.pickle', 'wb') as f:
        pickle.dump(real_m, f)

    with open('correctedM.pickle', 'wb') as f:
        pickle.dump(corrected_m, f)

    with open('realR.pickle', 'wb') as f:
        pickle.dump(real_mr, f)

    with open('correctedR.pickle', 'wb') as f:
        pickle.dump(corrected_mr, f)


if __name__ == "__main__":
    main()