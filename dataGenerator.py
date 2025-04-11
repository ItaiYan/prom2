import numpy as np
import pickle
import time
from tqdm import tqdm  # For progress bar
from hitLocations import run_parallel_rk45
from scipy.interpolate import RegularGridInterpolator

# Define the example function (you can modify this to your needs)

def create_points_file():
    # Generate parameter ranges
    n_points = 200
    v0_values = np.linspace(1100, 2450, n_points)  # Example range: -10 to 10
    theta_values = np.linspace(-np.pi/2, np.pi/2, n_points)
    mass_values = np.linspace(0.00001, 3, n_points)
    for i in range(n_points):
        v0_values[i] = round(v0_values[i], 6)
        theta_values[i] = round(theta_values[i], 6)
        mass_values[i] = round(mass_values[i], 6)
    # Create a dictionary to store results
    # Create a meshgrid of all combinations
    X, Y, Z = np.meshgrid(v0_values, theta_values, mass_values, indexing='ij')

    # Stack and reshape into a single list of (x, y, z) tuples
    combinations = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    print ("computing function")
    result = run_parallel_rk45(combinations)

    print ("converting to dictionery")
    results_dict = {}
    for i in tqdm(range(len(combinations))):
        results_dict[tuple(combinations[i])] = result[i]


    print ("saving the file")
    t = time.time()
    # Save to pickle file
    with open('function_values.pkl', 'wb') as f:
        pickle.dump({
            'results': results_dict,
            'x_range': v0_values,
            'y_range': theta_values,
            'z_range': mass_values
        }, f)
    e = time.time()
    print (f"it took {e-t} seconds to save the file")
    print(f"File saved as 'function_values.pkl'")

def load_file(pickle_file='function_values.pkl'):
    with open('function_values_array.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['arr'], data['x_range'], data['y_range'], data['z_range']
# Example of how to load and use the data
def get_interpolated_value(fragments, file):
    """
    Given a point (x, y, z), returns an interpolated function value using
    trilinear interpolation from the 8 closest grid points.

    Args:
        x, y, z (float): Coordinates of the input point
        pickle_file (str): Path to the pickle file from previous computation

    Returns:
        float: Interpolated function value at (x, y, z)
    """

    values, x, y, z = file
    interp_func = RegularGridInterpolator((x, y, z), values, bounds_error=False,
                                          fill_value=None)
    fragments = np.array(fragments)
    try:
        interpolated_values = interp_func(fragments[:, [0, 1, 3]])
    except:
        print("Interpolation error: Check if input points are within bounds.")
        pts = fragments[:, [0, 1, 3]]
        xb, yb, zb = (x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1])
        mask = ((pts[:, 0] < xb[0]) | (pts[:, 0] > xb[1]) |
                (pts[:, 1] < yb[0]) | (pts[:, 1] > yb[1]) |
                (pts[:, 2] < zb[0]) | (pts[:, 2] > zb[1]))
        oob = np.where(mask)[0]
        if oob.size:
            print(f"{oob.size} out-of-bounds fragment(s):")
            for i in oob: print(f"  idx {i}: {pts[i]}")
        raise ValueError("Interpolation error: Check if input points are within bounds.")
    angles = fragments[:, 2]
    magnitudes = interpolated_values[:, 0]
    current_x = np.cos(angles) * magnitudes
    current_y = np.sin(angles) * magnitudes
    mass = fragments[:, 3]

    res = np.column_stack((
        current_x,  # x_position
        current_y,  # y_position
        interpolated_values[:, 1],  # speed
        interpolated_values[:, 2],  # angle
        mass  # mass
    ))
    return res
    ############################################################
    results, x_range, y_range, z_range = file

    res = []
    for i in range(len(fragments)):
        v0 = fragments[i][0]
        theta = fragments[i][1]
        mass = fragments[i][3]

        # Find the nearest grid point indices (lower bound)
        x_idx = np.searchsorted(x_range, v0) - 1
        y_idx = np.searchsorted(y_range, theta) - 1
        z_idx = np.searchsorted(z_range, mass) - 1

        # Get grid dimensions
        n_points = len(x_range)

        # Ensure indices are within bounds
        x_idx = max(0, min(x_idx, n_points - 2))
        y_idx = max(0, min(y_idx, n_points - 2))
        z_idx = max(0, min(z_idx, n_points - 2))

        # Get the 8 cube vertices
        x0, x1 = x_range[x_idx], x_range[x_idx + 1]
        y0, y1 = y_range[y_idx], y_range[y_idx + 1]
        z0, z1 = z_range[z_idx], z_range[z_idx + 1]

        # Get function values at the 8 vertices
        points = {}
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    key = (round(x0 + dx * (x1 - x0), 6),
                           round(y0 + dy * (y1 - y0), 6),
                           round(z0 + dz * (z1 - z0), 6))
                    points[(dx, dy, dz)] = results.get(key, 0)  # Default to 0 if missing

        # Calculate normalized distances (dx, dy, dz between 0 and 1)
        dx = (v0 - x0) / (x1 - x0) if x1 != x0 else 0
        dy = (theta - y0) / (y1 - y0) if y1 != y0 else 0
        dz = (mass - z0) / (z1 - z0) if z1 != z0 else 0

        # Clamp values to [0, 1] to handle edge cases
        dx = max(0, min(1, dx))
        dy = max(0, min(1, dy))
        dz = max(0, min(1, dz))

        # Trilinear interpolation
        interpolated_value = (
                points[(0, 0, 0)] * (1 - dx) * (1 - dy) * (1 - dz) +
                points[(1, 0, 0)] * dx * (1 - dy) * (1 - dz) +
                points[(0, 1, 0)] * (1 - dx) * dy * (1 - dz) +
                points[(1, 1, 0)] * dx * dy * (1 - dz) +
                points[(0, 0, 1)] * (1 - dx) * (1 - dy) * dz +
                points[(1, 0, 1)] * dx * (1 - dy) * dz +
                points[(0, 1, 1)] * (1 - dx) * dy * dz +
                points[(1, 1, 1)] * dx * dy * dz
        )

        # Append the interpolated value to the results
        current_x = np.cos(fragments[i][2]) * interpolated_value[0]
        current_y = np.sin(fragments[i][2]) * interpolated_value[0]

        # Data structure: (x_position, y_position, speed, angle, mass)
        res.append((current_x, current_y, interpolated_value[1],
                    interpolated_value[2], mass))

    return res

# # Test the lookup
# test_x, test_y, test_z = -9.5, -7, -2.713
# value = get_interpolated_value(test_x, test_y, test_z)
# print(f"f({test_x}, {test_y}, {test_z}) = {value}")

if __name__ == "__main__":
    create_points_file()
    #print (get_interpolated_value([(2000, np.radians(45), 0.1)]))