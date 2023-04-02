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

    trajectory_np = np.array(trajectory)

    return trajectory_np


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

def generate_circle_trajectory(center, radius, num_points):
    # Generate an array of angles
    angles = np.linspace(0, 2*np.pi, num_points)
    
    # Calculate the x and y coordinates of each point on the circle
    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)
    
    # Combine the x and y coordinates into a single array
    trajectory = np.column_stack((x_coords, y_coords))
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