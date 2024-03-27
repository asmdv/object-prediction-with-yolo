import numpy as np
import cv2

# Define the Kalman Filter
kalman = cv2.KalmanFilter(2, 1)

# Initialize the transition matrix
kalman.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)

# Initialize the measurement matrix
kalman.measurementMatrix = np.array([[1, 0]], dtype=np.float32)

# Initialize the process noise covariance matrix
kalman.processNoiseCov = 0.0001 * np.eye(2, dtype=np.float32)

# Initialize the measurement noise covariance matrix
kalman.measurementNoiseCov = 0.1 * np.eye(1, dtype=np.float32)

# Initialize the state estimate
kalman.statePre = np.array([[0], [0]], dtype=np.float32)

# Initialize the covariance matrix
kalman.errorCovPre = np.eye(2, dtype=np.float32)

# Simulate measurements
measurements = [1, 2, 3, 4, 5]

# Perform Kalman filtering
for z in measurements:
    prediction = kalman.predict()
    measurement = np.array([[z]], dtype=np.float32)
    kalman.correct(measurement)
    print("Predicted:", prediction.flatten())
    print("Corrected:", kalman.statePost.flatten())
