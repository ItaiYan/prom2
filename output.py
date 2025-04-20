# this moodle will convert the output grid to a scatter plot of the hit
# positions of the shreds
import time
from datetime import datetime, timedelta
import matplotlib
import typing

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np
import math
import particleGenerator
import hitLocations
from numba import njit
from typing import List
from tqdm import tqdm


import dataGenerator

# constants for kill probability
C = 1
A = 3.4423 * (10 ** -3)
B = 23500
N = 0.41845

# constans for the fragment list
MASS = 0
VELOCITY = 1
THETA = 2
NUM_OF_RUNS = 10000
GRID_SIZE = 10

def main(shreds):
    x = shreds[:, 0]  # x coordinates
    y = shreds[:, 1]  # y coordinates
    masses = (shreds[:, 4]) * 15  # masses that will
    # determine
    # point sizes

    plt.figure(figsize=(10, 6))  # Optional: sets figure size
    scatter = plt.scatter(x, y,
                          s=masses,  # size of points based on mass
                          alpha=0.5)  # transparency (0-1)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot with Mass-based Point Sizes')
    plt.grid(alpha=0.5)
    plt.axis('equal')
    # Optional: Add a colorbar if you want to show mass scale
    # plt.colorbar(scatter, label='Mass')

    plt.show()








################################################################################################################# #
###############################################################################################################
################################################################################################################

def convert_to_table(shreds, squre_size=10, length=2000):
    """
    orgenizes the shreds in squares
    :param shreds:
    :param squre_size:
    :param length:
    :return:
    """
    A = np.empty(((length * 2) // squre_size, (length * 2) // squre_size), dtype=object)

    for i in range(((length * 2) // squre_size)):
        for j in range(((length * 2) // squre_size)):
            A[i, j] = []

    counter = 0
    for i in range(len(shreds)):
        if -length <= shreds[i][0] < length and -length <= shreds[i][1] < length:
            w = int((shreds[i][0] + length) / squre_size)
            h = int((shreds[i][1] + length) / squre_size)
            TVM = [shreds[i][3], shreds[i][2], shreds[i][4]] #theata, velocity, mass
            A[h, w].append(TVM)
        else:
            counter += 1

    for y in range(int(length / squre_size)):
        for x in range(len(A[0])):
            A[length // squre_size - y, x] = A[length // squre_size + y, x]

    return A


def color(data, scatter_x=None, scatter_y=None, table_size=1000):
    # Convert data to numpy array if it isn't already
    data = np.array(data)

    for row in range(len(data)):
        for column in range(len(data[0])):
            prob = data[row][column]
            if prob == 0:
                data[row][column] = -6
            else:
                data[row][column] = np.log10(prob)


    # Each square is 10 meters, so create coordinate arrays in meters
    x = np.arange(0, data.shape[1] + 1) * 10 - 2000  # +1 for pcolormesh edges
    y = np.arange(0, data.shape[0] + 1) * 10  - 2000# +1 for pcolormesh edges

    # Define a custom colormap: white (0) to red (1)
    colors = [(1, 1, 1), (1, 0, 0)]  # White to Red
    custom_cmap = LinearSegmentedColormap.from_list('white_to_red', colors,
                                                    N=256)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(x, y, data, cmap=custom_cmap, shading='auto')
    plt.colorbar(label='Value')
    plt.title('2D Grid: 0 (White) to 1 (Red)')

    # Add the scatter plot if scatter data is provided
    if scatter_x is not None and scatter_y is not None:
        plt.scatter(scatter_x, scatter_y, c='black', s=1,
                    label='Scatter Points')
        plt.legend()

    # Add contour line at z = -3.5
    x_centers = (np.arange(data.shape[1]) + 0.5) * 10 - 2000
    y_centers = (np.arange(data.shape[0]) + 0.5) * 10 - 2000
    cs = plt.contour(x_centers, y_centers, data, levels=[-3.5], colors='black')
    plt.clabel(cs, inline=True, fontsize=10, fmt='%1.1f')

    for lim in [-3.5]:

        # Calculate the radius for the circle
        # Find all points where value > -3.5
        y_idx, x_idx = np.where(data > lim)
        x_points = (x_idx + 0.5) * 10  # Convert to meters (center of cells)
        y_points = (y_idx + 0.5) * 10  # Convert to meters (center of cells)

        # Calculate distances from center (1000, 1000)
        center_x, center_y = 2000, 2000
        distances = np.sqrt((x_points - center_x) ** 2 + (y_points - center_y) ** 2)

        # Get maximum distance (radius) plus a small buffer
        radius = np.max(distances) if len(distances) > 0 else 0
        print (radius)
        radius += 10  # Add small buffer to ensure all points are enclosed

        # Add circle centered at (1000, 1000)
        circle = Circle((0, 0), radius, fill=False,
                        color='blue',
                        linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)

    # Adjust ticks to show meters, limiting to ~10 ticks for readability
    max_ticks = 10
    if data.shape[1] > max_ticks:
        x_step = (data.shape[1] * 10) // max_ticks  # Step in meters
        plt.xticks(np.arange(0, data.shape[1] * 10 + 1, x_step) - 2000)
    else:
        plt.xticks(x)

    if data.shape[0] > max_ticks:
        y_step = (data.shape[0] * 10) // max_ticks  # Step in meters
        plt.yticks(np.arange(0, data.shape[0] * 10 + 1, y_step) - 2000)
    else:
        plt.yticks(y)

    # Label axes in meters
    plt.xlabel('Distance (meters)')
    plt.ylabel('Distance (meters)')

    # Improve tick label readability
    plt.tick_params(axis='both', labelsize=8)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.show()

@njit
def _kill_probability_per_fragment_vec(theta: np.ndarray,
                                       velocity: np.ndarray,
                                       mass: np.ndarray) -> np.ndarray:
    """
    Numba-compiled vectorized fragment kill probability computation.
    Assumes all inputs are 1D NumPy arrays of the same length.
    """
    n = len(theta)
    probs = np.empty(n)
    for i in range(n):
        sin_theta = np.sin(theta[i])
        if sin_theta == 0:
            probs[i] = 0
            continue
        p_hit = (0.43 * np.cos(theta[i]) + 0.07 * sin_theta) / (GRID_SIZE**2 * sin_theta)
        p_hit = min(max(p_hit, 0), 1)

        mass_in_grain = mass[i] * 15432.3584
        velocity_fps = velocity[i] * 3.28084

        energy_term = mass_in_grain * velocity_fps**1.5 - B

        if energy_term <= 0.001:
            p_kill = 0.0
        else:
            p_kill = 1.0 - np.exp(-A * energy_term**N)
            p_kill = min(max(p_kill, 0), 1)

        probs[i] = p_hit * p_kill
    return probs

@njit
def _find_kill_probability_per_square_numba(fragments: np.ndarray) -> float:
    """
    JIT-accelerated per-square kill probability calculation.
    Input: (n, 5) NumPy array.
    """
    if fragments.shape[0] == 0:
        return -6.0

    theta = fragments[:, 0]
    velocity = fragments[:, 1]
    mass = fragments[:, 2]

    kill_probs = _kill_probability_per_fragment_vec(theta, velocity, mass)

    survival_probs = 1.0 - kill_probs
    prod_survival = 1.0
    for i in range(len(survival_probs)):
        prod_survival *= survival_probs[i]
    square_prob = 1.0 - prod_survival

    return square_prob

# -------------------- Python Interface --------------------

def find_kill_probability_per_square(fragments: List[np.ndarray]) -> float:
    """
    Wrapper for the Numba-compiled square probability function.
    Input: list of shape-(5,) arrays → converted to (n, 5) array.
    """
    if len(fragments) == 0:
        return 0
    fragments_np = np.array(fragments)
    return _find_kill_probability_per_square_numba(fragments_np)

def get_death_probability_map(hit_map: List[List[List[np.ndarray]]]) -> np.ndarray:
    """
    Compute log-kill-probability per square in the grid.
    Input: hit_map[x][y] = list of fragments (each is shape-(5,) np.ndarray)
    Returns: 2D NumPy array (X, Y)
    """
    x_dim = len(hit_map)
    y_dim = len(hit_map[0])
    result = np.empty((x_dim, y_dim))

    for x in range(x_dim):
        for y in range(y_dim):
            result[x, y] = find_kill_probability_per_square(hit_map[x][y])
    return result


if __name__ == '__main__':
    num_runs = 100
    death_prob = 0

    print ("before loading file")
    file = dataGenerator.load_file()
    print ("after loading file")
    # t = time.time()
    for i in tqdm(range(num_runs)):
        # generates initial conditions
        initial_conds = particleGenerator.test() # 0.01

        # generates a list of
        shreds = dataGenerator.get_interpolated_value(initial_conds, file) # 0.012


        hit_map = convert_to_table(shreds) #0.04-0.07

        if type(death_prob) == int: # 0.03
            death_prob = (np.array(get_death_probability_map(hit_map)) /
                          num_runs)
        else:
            death_prob += (np.array(get_death_probability_map(hit_map)) /
                           num_runs)

        # e = time.time()
        # future_time = datetime.now() + timedelta(seconds=((num_runs-i-1)*(e-t))/(i+1))
        # future_time = future_time.strftime("%H:%M:%S")
        # print (f"run {i+1} of {num_runs} average time {((e-t)/(i+1))} "
        #        f"seconds, eta {future_time}")
    color(death_prob)
