import numpy as np

def generate_linear_trajectory(start_point, end_point, num_points):
    # Calculate the step size in each dimension
    x_step = (end_point[0] - start_point[0]) / (num_points - 1)
    y_step = (end_point[1] - start_point[1]) / (num_points - 1)

    # Generate the trajectory
    trajectory = []
    for i in range(num_points):
        x = start_point[0] + (i * x_step)
        y = start_point[1] + (i * y_step)
        trajectory.append([x, y])

    return np.array(trajectory)


def generate_curved_trajectory(start_point, end_point, num_points):
    # Define the control point
    control_point = ((start_point[0] + end_point[0])/2, (start_point[1] + end_point[1])/2 + 2)

    # Create an array of equally spaced values from 0 to 1
    t = np.linspace(0, 1, num_points)

    # Define the function that maps t to x and y coordinates
    def f(t):
        return (1-t)**2*start_point[0] + 2*(1-t)*t*control_point[0] + t**2*end_point[0],\
               (1-t)**2*start_point[1] + 2*(1-t)*t*control_point[1] + t**2*end_point[1]

    # Map t to x and y coordinates using the function f
    trajectory = np.array([f(i) for i in t])

    return trajectory

def generate_circle_trajectory(center, radius, num_points):
    # Generate an array of angles
    angles = np.linspace(0, 2*np.pi, num_points)
    
    # Calculate the x and y coordinates of each point on the circle
    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)
    
    # Combine the x and y coordinates into a single array
    trajectory = np.column_stack((x_coords, y_coords))
    
    return trajectory

def generate_sin_wave_trajectory(length=1000, amplitude=100, period=100, wavelength=500, num_points=1000):
    # Calculate the distance of each point along the trajectory
    distances = np.linspace(0, length, num_points)

    # Calculate the wavelength if not provided
    if wavelength is None:
        wavelength = period / (2*np.pi)

    # Calculate the x and y coordinates of the trajectory
    x = distances
    y = amplitude * np.sin(2*np.pi/wavelength * distances)

    trajectory = np.column_stack((x, y))

    # Return the trajectory
    return trajectory



class KalmanFilter:
    def __init__(self, observation, use_acceleration=False):

        """
        Variables:
        - observation
        - use_acceleration

        Description:
            The variable 'observation' represents an observation in a 2-dimensional space.
            The variable 'use_acceleration' is a boolean variable used to determine if acceleration should be used in in process
            model.

        Attributes:
        - observation:
            - Shape: (2,)
            - Dtype: float64

        - use_acceleration:
            - Type: bool
        """

        self.dims = 2
        self.process_noise = 0.001
        self.measurement_noise = 0.05 
        dt=0.1

        ################################
        ### Kalman Filter Attributes ###
        ################################


        if use_acceleration:
            self.x = np.array([observation, [0., 0.], [0., 0.]]).flatten()
            self.P = np.vstack((
                                np.hstack((200*np.eye(self.dims), np.zeros((self.dims, 2*self.dims)))),
                                np.hstack((np.zeros((self.dims, self.dims)), 200*np.eye(self.dims), np.zeros((self.dims, self.dims)) )),
                                np.hstack((np.zeros((self.dims, 2*self.dims)), 1*np.eye(self.dims) )),
                                ))
            self.H = np.hstack((np.eye(self.dims), np.zeros((self.dims, 2*self.dims))))
            self.R = self.measurement_noise * np.eye(self.dims)
            self.F = np.vstack((
                                np.hstack((np.eye(self.dims), dt*np.eye(self.dims), (dt**2)/2*np.eye(self.dims))),
                                np.hstack((np.zeros((self.dims, self.dims)), np.eye(self.dims), dt*np.eye(self.dims))),
                                np.hstack((np.zeros((self.dims, 2 * self.dims)), np.eye(self.dims)))
                                ))
            self.Q = self.process_noise * np.eye(3*self.dims)

        else:
            self.x = np.array([observation, [0., 0.]]).flatten()
            self.P = np.vstack((
                                np.hstack((200*np.eye(self.dims), np.zeros((self.dims, self.dims)))),
                                np.hstack((np.zeros((self.dims, self.dims)), 200*np.eye(self.dims) )),
                                ))
            self.H = np.hstack((np.eye(self.dims), np.zeros((self.dims, self.dims))))
            self.F = np.vstack((
                                np.hstack((np.eye(self.dims), dt*np.eye(self.dims))),
                                np.hstack((np.zeros((self.dims, self.dims)), np.eye(self.dims))),
                                ))
            self.Q = self.process_noise * np.eye(2*self.dims)


        self.R = np.diag((self.dims)*[self.measurement_noise])


    def update(self, observation):
        """
        Variables:
        - observation

        Description:
            Update function takes in the variable 'observation' which represents an observation in a 2-dimensional space. 
            The function updates the state of the Kalman filter using latest observation.

        Attributes:
        - observation:
            - Shape: (2,)
            - Dtype: float64
        """

        ############################
        ### Kalman Filter Update ###
        ############################

        # calculate error or in KF terms 'y' --> difference between state and measurement
        z = observation
        err = z - np.matmul(self.H, self.x)
        # Get system uncertainty in measurement space
        S = np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R
        S_inv = np.linalg.inv(S)
        
        # Update Kalman gain
        K = np.matmul(np.matmul(self.P, self.H.transpose()), S_inv)
        # Update state with system cov
        self.x = self.x + np.matmul(K, err)
        self.P = np.matmul(np.eye(self.x.shape[0]) - np.matmul(K, self.H), self.P)

    def predict(self):
        """
        Description:
            Performs the prediction step of the Kalman filter.
        """

        #############################
        ### Kalman Filter Predict ###
        #############################

        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.transpose()) + self.Q

    def get_state(self):
        """
        Description:
            Returns the current state of the Kalman filter.
            
            Returns:
            - state: The current state of the Kalman filter.
            
            
            - Shape: (4,)
            - Dtype: float64
        """

        #########################
        ### Get Current State ###
        #########################
        return self.x
