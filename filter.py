import numpy as np
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from trajectories import simulate, simulate_rotation, get_view_vector

angles = np.arange(360)
radius = 1.5
views = []
positions = []
for angle in angles:
    pos = simulate(angle, radius)
    rot = simulate_rotation(angle, radius)
    view = get_view_vector(rot, pos)
    views.append(view)
    positions.append(pos)
positions = np.array(positions)
views = np.array(views)

my_filter = KalmanFilter(dim_x=6, dim_z=3)
dt = 1
my_filter.x = [positions[0][0], positions[0][1], positions[0][2], 0., 0., 0.]

# x, y, z, x', y', z',
my_filter.F = np.array([[1., 0, 0., dt, 0, 0],
                        [0., 1., 0, 0, dt, 0],
                        [0., 0., 1., 0., 0, dt],
                        [0., 0., 0., 1., 0., 0],
                        [0., 0., 0., 0., 1., 0],
                        [0., 0., 0, 0, 0, 1.]
                        ])    # state transition matrix

my_filter.H = np.array([
    [1., 0., 0., 0, 0, 0],
    [0., 1., 0., 0, 0, 0],
    [0., 0., 1., 0, 0, 0],
])    # Measurement function
# my_filter.P                  # covariance matrix
my_filter.R = 0.001                      # state uncertainty
my_filter.Q = Q_discrete_white_noise(3, dt, .1, 2) # process uncertainty


filtered = []
for i in range(1, len(positions)):
    my_filter.predict()
    y = np.array(positions[i]).reshape((3, 1))
    print(y.shape)
    my_filter.update(y)

    # do something with the output
    x = my_filter.x
    filtered.append(x)