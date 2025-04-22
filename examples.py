import matplotlib.pyplot as plt
import numpy as np
import math
import particleGenerator
import dataGenerator
from scipy.interpolate import RegularGridInterpolator
from output import *
from tqdm import tqdm
import hitLocations

def monte_carlo(num_of_runs = 20):
    kill_probability_per_r = 0
    for i in tqdm(range(num_of_runs)):
        if type(kill_probability_per_r) is int:
            kill_probability_per_r = generate_plot_temp_func(mean_mass=0.1,
                                                             v=2000,
                                                             angle_range=(
                                                                 np.radians(
                                                                     30),
                                                                 np.radians(
                                                                     30.01)),
                                                             num_of_fragments=1000, r_max=1000)
        else:
            kill_probability_per_r += generate_plot_temp_func(mean_mass=0.1,
                                                               v=2000,
                                                               angle_range=(
                                                                   np.radians(30), np.radians(30.01)),
                                                               num_of_fragments=1000, r_max=1000)

    kill_probability_per_r /= num_of_runs

    for i in range(len(kill_probability_per_r)):
        if kill_probability_per_r[i] < 10 ** -6:
            kill_probability_per_r[i] = -6
        else:
            kill_probability_per_r[i] = np.log10(kill_probability_per_r[i])
    r_max = 3000
    # plot bin_prob
    # Create a scatter plot
    fig, ax = plt.subplots()
    x = range(len(kill_probability_per_r))
    y = kill_probability_per_r
    ax.scatter(x, y)
    ax.set_xlabel('Distance from bomb center (m)')
    ax.set_ylabel('Probability of death')
    ax.set_title('Death Probability vs Distance from Bomb Center')
    ax.set_xlim(0, r_max)
    ax.set_ylim(-6, 0)
    ax.grid(alpha=0.5)
    plt.show()

def generate_plot_temp_func(mean_mass, v, angle_range, num_of_fragments,
                            r_max = 1000):
    """
    generate a plot shwoin the deth probabilitt based on the distance from
    the bomb center.
    Args:
        mean_m (float): mean mass of the fragment
        v (float): velocity of the fragment
        angle_range (tuple): range of angles in degrees (min, max)
        r_max (int): maximum radius to consider for the plot
    """

    # generate all of the particels
    fragments = []
    mass_accum = 0
    mass_target = 2 * mean_mass * num_of_fragments

    dphi = angle_range[1] - angle_range[0]
    theta_min = angle_range[0]
    theta_max = angle_range[1]
    d = 7800  # density of the bomb

    while mass_accum < mass_target:
        batch_size = 500  # process in chunks to reduce overhead
        theta = np.random.uniform(theta_min, theta_max, batch_size)
        mass = particleGenerator.generate_mass_vectorized(mean_mass, batch_size)
        velocity = np.full(batch_size, v)
        density = np.full(batch_size, d)
        fi = np.zeros(batch_size)

        new_fragments = np.stack([velocity, theta, fi, mass, density],
                                 axis=-1)

        total_mass = np.cumsum(mass)
        stop_idx = np.searchsorted(mass_accum + total_mass, mass_target)

        if stop_idx < batch_size:
            fragments.append(new_fragments[:stop_idx + 1])
            mass_accum += total_mass[stop_idx]
            break
        else:
            fragments.append(new_fragments)
            mass_accum += total_mass[-1]

    # Concatenate all fragments into a single array
    # v, theta, fi, mass, density
    section_all = np.concatenate(fragments, axis=0)

    #file = dataGenerator.load_file()

    # values, x, y, z = file
    # interp_func = RegularGridInterpolator((x, y, z), values, bounds_error=False,
    #                                       fill_value=None)
    # section_all = np.array(section_all)
    # interpolated_values = interp_func(section_all[:, [0, 1, 3]]) #r, v, theta
    # interpolated_values = np.column_stack((interpolated_values, section_all[:, 3]))
    interpolated_values = hitLocations.main(section_all)
    interpolated_values = interpolated_values[:, [0, 2, 3, 4]]  # r, v, theta, mass



    min_expected_x_int = 0  # Includes x values down to 0.00...
    max_expected_x_int = 3300  # Includes x values up to 1000.99...
    num_bins = max_expected_x_int - min_expected_x_int + 1
    offset = min_expected_x_int  # Index 0 corresponds to x = min_expected_x_int
    # --------------------------------

    # 1. Create an empty NumPy array of the correct size with dtype=object
    binned_fragments = np.empty(num_bins, dtype=object)

    # 2. Initialize each element explicitly with an empty Python list
    for i in range(num_bins):
        binned_fragments[i] = []  # Assigns a new list to each slot

    # Bin the fragments (will error if x is outside expected range)
    for fragment in interpolated_values:
        bin_index = math.floor(fragment[0]) - offset
        # No bounds check here - relies on input being within range
        binned_fragments[bin_index].append(fragment)

    bin_probabilities = []
    for bin in binned_fragments:
        bin_prob = 1
        for fragment in bin:
            bin_prob *= (1-p_kill(fragment[3], fragment[1]) * p_hit(fragment[2],
                                                                 fragment[0]))
        bin_prob = 1 - bin_prob
        bin_probabilities.append(bin_prob)
    return np.array(bin_probabilities)




def p_hit(theta, r):
    sin_theta = np.sin(theta)
    if sin_theta == 0:
        return 0
    return min((0.43 * np.cos(theta) + 0.07 * sin_theta) / (r * sin_theta), 1)

def p_kill(mass, velocity):
    mass_in_grain = mass * 15432.3584
    velocity_fps = velocity * 3.28084

    energy_term = mass_in_grain * velocity_fps ** 1.5 - B

    if energy_term <= 0.001:
        p_kill = 0.0
    else:
        p_kill = 1.0 - np.exp(-A * energy_term ** N)
        p_kill = min(max(p_kill, 0), 1)
    return p_kill


if __name__ == "__main__":
    monte_carlo()



