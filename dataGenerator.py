import numpy as np
import pickle
import time
from tqdm import tqdm  # For progress bar
from oldSimulaion import run_parallel_rk45
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
def get_interpolated_value(fragments, interp_func):
    """
    Given a point (x, y, z), returns an interpolated function value using
    trilinear interpolation from the 8 closest grid points.

    Args:
        x, y, z (float): Coordinates of the input point
        pickle_file (str): Path to the pickle file from previous computation

    Returns:
        float: Interpolated function value at (x, y, z)
    """
    #
    # values, x, y, z = file
    # interp_func = RegularGridInterpolator((x, y, z), values, bounds_error=False,
    #                                       method='cubic', fill_value=None)

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

# # Test the lookup
# test_x, test_y, test_z = -9.5, -7, -2.713
# value = get_interpolated_value(test_x, test_y, test_z)
# print(f"f({test_x}, {test_y}, {test_z}) = {value}")

if __name__ == "__main__":
    create_points_file()
    #print (get_interpolated_value([(2000, np.radians(45), 0.1)]))