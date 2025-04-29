# this moodle will convert the output grid to a scatter plot of the hit
# positions of the shreds
import os
import time
from datetime import datetime, timedelta
import matplotlib
import typing
import openAreaSimulation
import urbanAreaSimulation

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
import numpy as np
import math
import particleGenerator
import oldSimulaion
from numba import njit
from typing import List
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import json

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

    for y in range(int(length / squre_size)):
        for x in range(len(A[0])):
            A[length // squre_size - y, x] = A[length // squre_size + y, x]

    return A


def color(data, building_height_grid=None, scatter_x=None, scatter_y=None,
          grid_extent_plot_m=2000, output_filename="plots.png"):
    """
    Generates a heatmap plot with optional scatter points and potentially
    higher-resolution building height overlay, and saves it to a file.

    Args:
        data (np.ndarray): 2D array of values (e.g., log probability) for the heatmap.
        building_height_grid (np.ndarray, optional): 2D array of building heights.
                                                     Can have a different resolution than 'data',
                                                     but should cover the same physical extent.
                                                     Values > 0 indicate building presence.
                                                     Defaults to None.
        scatter_x (np.ndarray, optional): X-coordinates for scatter points. Defaults to None.
        scatter_y (np.ndarray, optional): Y-coordinates for scatter points. Defaults to None.
        grid_extent_plot_m (int, optional): The total physical extent (width/height in meters)
                                            covered by both grids, assumed centered at 0,0.
                                            Defaults to 2000.
        output_filename (str, optional): Path where the plot image will be saved.
                                         Defaults to "plot.png".
    """
    # --- Input Validation ---
    data = np.array(data)
    if data.ndim != 2 or data.size == 0:
        print("Error: Input data for heatmap is not a valid 2D array.")
        return

    # --- Data Transformation (Log Scale for Heatmap) ---
    plot_data = data.copy().astype(float)
    threshold = 1e-6
    plot_data[plot_data < threshold] = threshold
    with np.errstate(divide='ignore'): # Ignore log10(0) warnings, handled next
        plot_data = np.log10(plot_data)
    plot_data[data < threshold] = -6.0 # Set original sub-threshold values to -6


    # --- Coordinate Setup ---
    heatmap_height_cells, heatmap_width_cells = plot_data.shape
    if heatmap_width_cells <= 0 or heatmap_height_cells <= 0:
        print("Error: Heatmap data grid has zero dimensions.")
        return
    heatmap_cell_size_x = grid_extent_plot_m / heatmap_width_cells
    heatmap_cell_size_y = grid_extent_plot_m / heatmap_height_cells
    plot_origin_x = -grid_extent_plot_m / 2
    plot_origin_y = -grid_extent_plot_m / 2
    plot_extent = [plot_origin_x, plot_origin_x + grid_extent_plot_m,
                   plot_origin_y, plot_origin_y + grid_extent_plot_m]
    x_edges = np.linspace(plot_extent[0], plot_extent[1], heatmap_width_cells + 1)
    y_edges = np.linspace(plot_extent[2], plot_extent[3], heatmap_height_cells + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(11, 10)) # Get figure and axes objects

    # Define colormap for heatmap
    heatmap_colors = [(1, 1, 1), (1, 0.8, 0.8), (1, 0, 0)]
    heatmap_norm = Normalize(vmin=-6, vmax=0)
    heatmap_cmap = LinearSegmentedColormap.from_list('white_to_red', heatmap_colors, N=256)

    # Create the heatmap
    mesh = ax.pcolormesh(x_edges, y_edges, plot_data, cmap=heatmap_cmap, norm=heatmap_norm, shading='flat')
    fig.colorbar(mesh, ax=ax, label='Log10(Kill Probability)', fraction=0.046, pad=0.04)
    ax.set_title('Kill Probability Heatmap with Buildings')
    ax.axis('equal')

    # --- Add Building Height Grid Overlay ---
    if building_height_grid is not None:
        building_height_grid = np.array(building_height_grid)
        if building_height_grid.ndim == 2 and building_height_grid.size > 0:
            print(f"Overlaying building height grid (shape: {building_height_grid.shape}).")
            masked_buildings = np.ma.masked_where(building_height_grid <= 0, building_height_grid)
            building_cmap = plt.cm.Greys
            building_cmap.set_bad(alpha=0)
            ax.imshow(masked_buildings, cmap=building_cmap, origin='lower', extent=plot_extent,
                      aspect='equal', alpha=0.5, interpolation='none')
        else:
            print("Warning: Building height grid is not a valid 2D array. Skipping overlay.")


    # Add scatter plot
    if scatter_x is not None and scatter_y is not None:
        ax.scatter(scatter_x, scatter_y, c='dimgray', s=1, alpha=0.5, label='Scatter Points')


    # Add contour line
    try:
        if plot_data.shape == (len(y_centers), len(x_centers)):
             cs = ax.contour(x_centers, y_centers, plot_data, levels=[-3.5], colors='blue', linewidths=1.5)
             ax.clabel(cs, inline=True, fontsize=9, fmt='%1.1f (logP)')
        else:
             print("Warning: Heatmap data dimensions mismatch for contour plot.")
    except Exception as e:
        print(f"Could not plot contour: {e}")


    # --- Circle Calculation ---
    contour_level_for_circle = -3.5
    y_idx_c, x_idx_c = np.where(plot_data > contour_level_for_circle)
    heatmap_cell_size_x = grid_extent_plot_m / plot_data.shape[1]
    heatmap_cell_size_y = grid_extent_plot_m / plot_data.shape[0]
    x_points_c = (x_idx_c + 0.5) * heatmap_cell_size_x + plot_origin_x
    y_points_c = (y_idx_c + 0.5) * heatmap_cell_size_y + plot_origin_y
    if x_points_c.size > 0:
        distances_c = np.sqrt(x_points_c**2 + y_points_c**2)
        radius_c = np.max(distances_c)
        print(f"Max distance for points > {contour_level_for_circle} (logP): {radius_c:.2f} m")
        circle = Circle((0, 0), radius_c, fill=False, color='magenta', linestyle=':', linewidth=2,
                        label=f'Radius for > {contour_level_for_circle} logP')
        ax.add_patch(circle)
    else:
        print(f"No data points found above contour level {contour_level_for_circle} for circle calculation.")


    # --- Final Touches ---
    ax.set_xlabel('Distance X (meters)')
    ax.set_ylabel('Distance Y (meters)')
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(alpha=0.4, linestyle=':')
    ax.set_xlim(plot_extent[0], plot_extent[1])
    ax.set_ylim(plot_extent[2], plot_extent[3])
    ax.legend()
    plt.tight_layout()

    # --- Save the plot instead of showing it ---
    bomb, theta_hit, velocity, urban_area = particleGenerator.open_data()
    output_filename = str(theta_hit) + "m"
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        plt.savefig(output_filename, dpi=300) # Save with high resolution
        print(f"Plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
    finally:
        # --- Close the plot to free memory ---
        plt.close(fig) # Close the figure associated with the axes 'ax'



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

def open_area_simulatioan(num_runs = 200):
    death_prob = 0

    # print ("before loading file")
    # file = dataGenerator.load_file()
    # print ("after loading file")
    # values, x, y, z = file
    # interp_func = RegularGridInterpolator((x, y, z), values, bounds_error=False,
    #                                       method='quintic', fill_value=None)
    # print ("after creating the interpolation function")

    # t = time.time()
    for i in tqdm(range(num_runs)):
        # generates initial conditions
        initial_conds = particleGenerator.test()  # 0.01

        # generates a list of
        shreds = openAreaSimulation.main(initial_conds)  # 0.012

        hit_map = convert_to_table(shreds)  # 0.04-0.07

        if type(death_prob) == int:  # 0.03
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


def urban_area_simulation(num_runs=200):
    death_prob = 0

    building_grid, grid_origin, grid_cell_size = urbanAreaSimulation.create_building_grid()
    for i in tqdm(range(num_runs)):
        # generates initial conditions
        initial_conds = particleGenerator.test()  # 0.01

        # generates a list of
        shreds = urbanAreaSimulation.run_particle_simulation(
            initial_conds, building_grid, grid_origin, grid_cell_size,
            initial_height=1.5, dt=0.05, t_max=45.0
        )

        hit_map = convert_to_table(shreds)  # 0.04-0.07

        if type(death_prob) == int:  # 0.03
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

if __name__ == '__main__':
    with open("input.JSON") as f:
        data = json.load(f)
        urban_area = bool(data["urban area"])

    if urban_area:
        urban_area_simulation()
    else:
        open_area_simulatioan(1000)



