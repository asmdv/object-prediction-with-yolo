from filterpy.kalman import KalmanFilter
import numpy as np
import time



# Create Kalman filter for 2D position
f = KalmanFilter(dim_x=2, dim_z=2)

# Initial state (position only)
f.x = np.array([[2.],  # x position
                 [5.]])  # y position

# State transition matrix (no velocity, stays at position)
f.F = np.array([[1., 0.],
                [0., 1.]])

# Measurement matrix (measuring x and y positions directly)
f.H = np.array([[1., 0.],
                [0., 1.]])

# Process and measurement noise covariances (adjust as needed)
f.P *= 1000.  # Initial uncertainty
f.R = np.array([[5., 0.],  # Measurement noise covariance
                [0., 5.]])
f.Q = np.eye(2) * 0.1  # Process noise covariance for position (no velocity)

# Simulate and filter
start_time = time.time()
for i in range(10):
    z = np.array([[i], [2*i]])  # Example measurements (x and y)
    print(f.predict())
    f.update(z)

print("--- %s seconds ---" % (time.time() - start_time))
