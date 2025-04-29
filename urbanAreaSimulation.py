import numpy as np
import numba  # Import Numba
import time
import matplotlib
import matplotlib.pyplot as plt  # Make sure matplotlib is imported
import cityStructure

matplotlib.use('TkAgg')

# Assuming particleGenerator.py exists and has a test() function
# that returns fragments like: [V, theta, phi, mass, density]
# If not, replace particleGenerator.test() with your actual data source
try:
    import particleGenerator
except ImportError:
    print("Warning: particleGenerator.py not found. Using placeholder data.")


    # Define a simple placeholder function if particleGenerator is missing
    def dummy_particle_generator(n_particles=100):
        fragments = np.zeros((n_particles, 5))
        fragments[:, 0] = np.random.uniform(500, 1500, n_particles)  # V
        fragments[:, 1] = np.random.uniform(0.1, np.pi / 2 - 0.1,
                                            n_particles)  # theta
        fragments[:, 2] = np.random.uniform(0, 2 * np.pi, n_particles)  # phi
        fragments[:, 3] = np.random.uniform(0.01, 1.0, n_particles)  # mass
        fragments[:, 4] = np.full(n_particles, 7800)  # density
        return fragments


    particleGenerator = type('obj', (object,),
                             {'test': dummy_particle_generator})()

# --- Constants for Return Status ---
IMPACT_STATUS_GROUND = 0
IMPACT_STATUS_BUILDING = 1
IMPACT_STATUS_TIMEOUT = 2


# --- Derivative Function (Assumed unchanged from your original) ---
@numba.jit(nopython=True, cache=True)
def projectile_deriv_numba(state, masses, areas, t):
    """
    Calculates the derivatives (velocities and accelerations) for projectiles.
    Args:
        state (np.ndarray): Current state [x_range, y_height, vx, vy] for M particles (M, 4).
        masses (np.ndarray): Masses for M particles (M,).
        areas (np.ndarray): Cross-sectional areas for M particles (M,).
        t (float): Current time (unused in this specific derivative calculation).
    Returns:
        np.ndarray: Derivatives [vx, vy, ax, ay] for M particles (M, 4).
    """
    rho = 1.225  # Air density (kg/m^3)
    C_D = 1.1  # Drag coefficient (example, could be per-particle)
    g = 9.81  # Gravity (m/s^2)

    # Unpack state variables for active particles
    vx = state[:,
         2]  # Velocity component in the original launch direction plane
    vy = state[:, 3]  # Vertical velocity component

    # Calculate speed in the 2D simulation plane (range vs height)
    velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)
    safe_velocity_magnitude = velocity_magnitude + 1e-9  # Avoid division by zero

    # Calculate magnitude of drag force
    drag_force_magnitude = 0.5 * rho * C_D * areas * velocity_magnitude ** 2

    # Calculate components of drag force
    drag_x = -drag_force_magnitude * (vx / safe_velocity_magnitude)
    drag_y = -drag_force_magnitude * (vy / safe_velocity_magnitude)

    # Calculate acceleration components
    ax = drag_x / masses
    ay = -g + (drag_y / masses)  # Gravity acts downwards

    # Return derivatives: [velocity_x, velocity_y, acceleration_x, acceleration_y]
    return np.column_stack((vx, vy, ax, ay))


# --- Core Simulation Function (MODIFIED for Building Collisions) ---
@numba.jit(nopython=True, cache=True)
def rk4_vectorized_simulation_numba(
        initial_state_full,
        # Now includes phi: [x, y, vx, vy, mass, phi, density, area, C_D]
        dt,
        t_max,
        building_grid,  # 2D Boolean array (True=building)
        grid_origin,  # Tuple/Array (min_x, min_y) of the grid
        grid_cell_size  # Float, size of each square grid cell
):
    """
    Performs a vectorized RK4 simulation for N particles until they hit the ground
    OR a building, optimized with Numba.

    Args:
        initial_state_full (np.ndarray): Shape (N, 9). Columns MUST be
                                         [x, y, vx, vy, mass, phi, density, area, C_D].
        dt (float): Time step.
        t_max (float): Maximum simulation time.
        building_grid (np.ndarray): 2D boolean array where True indicates a building.
        grid_origin (np.ndarray): 1D array/tuple like [min_x, min_y] of the grid.
        grid_cell_size (float): The width/height of a single grid cell.

    Returns:
        np.ndarray: Shape (N, 4). Columns:
                    [final_range, final_speed, final_angle_rad, impact_status]
    """
    n_particles = initial_state_full.shape[0]
    grid_height_cells = building_grid.shape[0]
    grid_width_cells = building_grid.shape[1]

    # --- Initialization ---
    # State for simulation: [range, height, v_range, v_height]
    state = np.zeros((n_particles, 4), dtype=np.float64)
    state[:, 0] = initial_state_full[:, 0]  # Initial range (usually 0)
    state[:, 1] = initial_state_full[:, 1]  # Initial height (y)
    # Initial horizontal speed component along phi direction
    initial_speed_horizontal = np.sqrt(
        initial_state_full[:, 2] ** 2)  # Simplified initial vx magnitude
    state[:, 2] = initial_speed_horizontal  # Initial v_range
    state[:, 3] = initial_state_full[:, 3]  # Initial v_height (vy)

    # Extract other properties needed for simulation
    masses = initial_state_full[:, 4].copy().astype(np.float64)
    phis = initial_state_full[:, 5].copy().astype(
        np.float64)  # Get initial phi angles
    areas = initial_state_full[:, 7].copy().astype(np.float64)  # Get areas

    t = 0.0

    # --- Result Storage ---
    final_results = np.full((n_particles, 4), np.nan, dtype=np.float64)
    final_results[:, 3] = IMPACT_STATUS_TIMEOUT  # Default status is Timeout

    # Boolean mask to track active particles
    active_mask = np.ones(n_particles, dtype=np.bool_)

    # --- Handle particles starting at or below ground ---
    for i in range(n_particles):
        if state[i, 1] <= 0:
            final_results[i, 0] = state[i, 0]
            final_results[i, 1] = np.sqrt(state[i, 2] ** 2 + state[i, 3] ** 2)
            final_results[i, 2] = np.arctan2(state[i, 3],
                                             state[i, 2]) if np.abs(
                state[i, 3]) > 1e-9 else 0.0
            final_results[i, 3] = IMPACT_STATUS_GROUND
            active_mask[i] = False

    # --- Simulation Loop ---
    while t < t_max:
        current_active_indices = np.where(active_mask)[0]
        num_active = len(current_active_indices)
        if num_active == 0: break

        state_active = state[current_active_indices]
        masses_active = masses[current_active_indices]
        areas_active = areas[current_active_indices]
        state_prev_active = state_active.copy()

        # --- RK4 Step ---
        k1 = projectile_deriv_numba(state_active, masses_active, areas_active,
                                    t) * dt
        k2 = projectile_deriv_numba(state_active + 0.5 * k1, masses_active,
                                    areas_active, t + 0.5 * dt) * dt
        k3 = projectile_deriv_numba(state_active + 0.5 * k2, masses_active,
                                    areas_active, t + 0.5 * dt) * dt
        k4 = projectile_deriv_numba(state_active + k3, masses_active,
                                    areas_active, t + dt) * dt
        state_active += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += dt
        state[current_active_indices] = state_active

        # --- Collision Checks ---
        for i in range(num_active):
            idx_global = current_active_indices[i]
            current_state = state_active[i]
            current_range, current_height = current_state[0], current_state[1]

            # --- 1. Building Collision Check ---
            current_phi = phis[idx_global]
            ground_x = current_range * np.cos(current_phi)
            ground_y = current_range * np.sin(current_phi)
            grid_x_float = (ground_x - grid_origin[0]) / grid_cell_size
            grid_y_float = (ground_y - grid_origin[1]) / grid_cell_size

            if 0 <= grid_x_float < grid_width_cells and 0 <= grid_y_float < grid_height_cells:
                grid_x_idx = int(np.floor(grid_x_float))
                grid_y_idx = int(np.floor(grid_y_float))
                if building_grid[grid_y_idx, grid_x_idx] > current_height > 1:
                    final_results[idx_global, 0] = current_range
                    final_results[idx_global, 1] = np.sqrt(
                        current_state[2] ** 2 + current_state[3] ** 2)
                    final_results[idx_global, 2] = np.arctan2(current_state[3],
                                                              current_state[
                                                                  2]) if np.abs(
                        current_state[3]) > 1e-9 else 0.0
                    final_results[idx_global, 3] = IMPACT_STATUS_BUILDING
                    active_mask[idx_global] = False
                    continue  # Skip ground check

            # --- 2. Ground Collision Check ---
            if current_height <= 0:
                y_prev, range_prev = state_prev_active[i, 1], state_prev_active[
                    i, 0]
                impact_range, impact_vx, impact_vy = current_range, \
                current_state[2], current_state[3]
                if y_prev > 0 and y_prev > current_height:  # Interpolate if crossed boundary
                    delta_y = y_prev - current_height
                    if delta_y > 1e-12:
                        alpha = max(0.0, min(1.0, y_prev / delta_y))
                        impact_range = range_prev + alpha * (
                                    current_range - range_prev)
                        impact_vx = state_prev_active[i, 2] + alpha * (
                                    current_state[2] - state_prev_active[i, 2])
                        impact_vy = state_prev_active[i, 3] + alpha * (
                                    current_state[3] - state_prev_active[i, 3])
                    else:  # Grazed ground
                        impact_range = range_prev
                        impact_vx = state_prev_active[i, 2]
                        impact_vy = state_prev_active[i, 3]

                final_results[idx_global, 0] = impact_range
                final_results[idx_global, 1] = np.sqrt(
                    impact_vx ** 2 + impact_vy ** 2)
                final_results[idx_global, 2] = np.arctan2(impact_vy,
                                                          impact_vx) if np.abs(
                    impact_vy) > 1e-9 else 0.0
                final_results[idx_global, 3] = IMPACT_STATUS_GROUND
                active_mask[idx_global] = False

    # --- Handle Timeouts ---
    timeout_indices = np.where(active_mask)[0]
    if len(timeout_indices) > 0:
        last_state_timeout = state[timeout_indices]
        final_results[timeout_indices, 0] = last_state_timeout[:, 0]
        final_results[timeout_indices, 1] = np.sqrt(
            last_state_timeout[:, 2] ** 2 + last_state_timeout[:, 3] ** 2)
        vy_timeout, vx_timeout = last_state_timeout[:, 3], last_state_timeout[:,
                                                           2]
        angles_timeout = np.arctan2(vy_timeout, vx_timeout)
        angles_timeout[np.abs(vy_timeout) <= 1e-9] = 0.0
        final_results[timeout_indices, 2] = angles_timeout
        # Status is already IMPACT_STATUS_TIMEOUT by default

    return final_results


# --- Particle Data Conversion ---
def particle_conversion_for_simulation(fragments, initial_height=1.0, ks=0.298,
                                       default_C_D=1.1):
    """
    Converts fragment data [V, theta, phi, mass, density] to the format needed
    by the Numba simulation function.
    Returns: np.ndarray: Shape (N, 9) -> [x0, y0, vx0, vy0, mass, phi, density, area, C_D]
    """
    v_idx, theta_idx, phi_idx, mass_idx, rho_idx = range(5)
    N = fragments.shape[0]
    x0 = np.zeros(N)
    y0 = np.full(N, initial_height)
    V = fragments[:, v_idx]
    theta_launch = fragments[:, theta_idx]
    vx0 = V * np.cos(theta_launch)
    vy0 = V * np.sin(theta_launch)
    mass = fragments[:, mass_idx]
    phi = fragments[:, phi_idx]
    density = fragments[:, rho_idx]
    base_area_calc = mass / (density * ks)
    base_area_calc[base_area_calc < 0] = 0
    areas = base_area_calc ** (2 / 3)
    cds = np.full(N, default_C_D)
    return np.column_stack((x0, y0, vx0, vy0, mass, phi, density, areas, cds))


# --- Main Runner Function ---
def run_particle_simulation(
        initial_conditions_fragments,  # Input is [V, theta, phi, mass, density]
        building_grid,
        grid_origin,
        grid_cell_size,
        initial_height=1.0,
        dt=0.01,
        t_max=30.0
):
    """
    Sets up and runs the Numba-optimized particle simulation.
    Returns: np.ndarray: Shape (N, 6). Columns:
             [final_x, final_y, final_speed, final_angle_rad, mass, impact_status]
    """
    # print(f"Preparing simulation for {initial_conditions_fragments.shape[
    # 0]} particles...")
    start_time = time.time()

    # 1. Convert fragment data
    initial_state_full = particle_conversion_for_simulation(
        initial_conditions_fragments, initial_height=initial_height
    )

    # 2. Run Core Simulation
    result_sim = rk4_vectorized_simulation_numba(
        initial_state_full, dt, t_max, building_grid,
        np.array(grid_origin, dtype=np.float64), grid_cell_size
    )

    # 3. Post-process results
    final_range, final_speed, final_angle, impact_status = result_sim[:,
                                                           0], result_sim[:,
                                                               1], result_sim[:,
                                                                   2], result_sim[
                                                                       :, 3]
    original_phi, original_mass = initial_state_full[:, 5], initial_state_full[
                                                            :, 4]
    final_x = final_range * np.cos(original_phi)
    final_y = final_range * np.sin(original_phi)
    final_x[np.isnan(final_range)] = np.nan
    final_y[np.isnan(final_range)] = np.nan

    final_output = np.column_stack((final_x, final_y, final_speed, final_angle,
                                    original_mass, impact_status))

    end_time = time.time()
    # print(f"Simulation finished in {end_time - start_time:.4f} seconds.")

    status_codes = final_output[:, 5]
    ground_mask = (status_codes == IMPACT_STATUS_GROUND)
    final_output = final_output[ground_mask]
    # ----------------------
    return final_output


# --- Building Grid Generation ---
def create_building_grid():
    """ Creates a boolean grid representing building footprints. """
    return cityStructure.get_building_matrix(), [-1000, -1000], 1.0


# --- Example Usage Block ---
if __name__ == "__main__":
    print("--- Running Simulation Example with Building Collisions ---")

    # --- 1. Define Grid and Buildings ---
    GRID_SIZE_CELLS = (2000, 2000)  # cells (y_rows, x_cols)
    GRID_EXTENT_METERS = (2000, 2000)  # meters (width, height), centered at 0,0
    # Note: 'length' key in example_buildings is not used by create_building_grid
    example_buildings = [
        {'x': -250, 'y': -150, 'width': 100, 'height': 200, 'length': 100},
        {'x': 100, 'y': 50, 'width': 150, 'height': 80, "length": 100},
        {'x': 500, 'y': 400, 'width': 60, 'height': 60, "length": 100},
    ]
    building_grid, grid_origin, grid_cell_size = create_building_grid()

    # Create an empty grid for the "no buildings" simulation
    empty_building_grid = np.zeros_like(building_grid)

    # Optional: Visualize the building grid (using only imshow)
    try:
        plt.figure(figsize=(8, 8))
        # extent=[left, right, bottom, top]
        extent = [grid_origin[0], grid_origin[0] + GRID_EXTENT_METERS[0],
                  grid_origin[1], grid_origin[1] + GRID_EXTENT_METERS[1]]
        plt.imshow(building_grid, origin='lower', extent=extent, cmap='gray',
                   aspect='equal')
        plt.title("Building Footprint Grid (True=Black)")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        # --- REMOVED the loop drawing red rectangles ---
        # for b in example_buildings:
        #      rect = plt.Rectangle((b['x'], b['y']), b['width'], b['height'], linewidth=1, edgecolor='r', facecolor='none')
        #      plt.gca().add_patch(rect)
        # -----------------------------------------------
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plt.show(block=False)
        plt.pause(1)  # Allow time for plot window to appear
    except ImportError:
        print("Matplotlib not found, skipping grid visualization.")
    except Exception as e:
        print(f"Error during grid plotting: {e}")

    # --- 2. Generate Particle Data ---
    N_PARTICLES_EXAMPLE = 5000
    # fragments format: [V, theta, phi, mass, density]
    # Make sure particleGenerator.test() provides N_PARTICLES_EXAMPLE
    try:
        fragments = particleGenerator.test(N_PARTICLES_EXAMPLE)
    except TypeError:  # Handle case where test() doesn't take an argument
        print(
            "Warning: particleGenerator.test() might not accept particle count. Generating default.")
        fragments = particleGenerator.test()

    if fragments.shape[0] == 0:
        print("Error: No fragments generated. Exiting.")
        exit()
    print(f"Generated {fragments.shape[0]} fragments.")

    # --- 3. Run Simulation WITH Buildings ---
    print("\n--- Running Simulation WITH Buildings ---")
    simulation_results_buildings = run_particle_simulation(
        fragments, building_grid, grid_origin, grid_cell_size,
        initial_height=1.5, dt=0.005, t_max=400.0
    )

    # --- 4. Run Simulation WITHOUT Buildings ---
    print("\n--- Running Simulation WITHOUT Buildings ---")
    simulation_results_no_buildings = run_particle_simulation(
        fragments, empty_building_grid, grid_origin, grid_cell_size,
        # Use empty grid
        initial_height=1.5, dt=0.05, t_max=45.0
    )

    # --- 5. Output and Plot Results ---
    # Output format: [final_x, final_y, final_speed, final_angle_rad, mass, impact_status]

    # --- Plot WITH Buildings ---
    if simulation_results_buildings is not None and \
            simulation_results_buildings.shape[0] > 0:
        print(
            f"\n--- Results WITH Buildings ({simulation_results_buildings.shape[0]} particles processed) ---")
        status_codes = simulation_results_buildings[:, 5]
        num_ground = np.sum(status_codes == IMPACT_STATUS_GROUND)
        num_building = np.sum(status_codes == IMPACT_STATUS_BUILDING)
        num_timeout = np.sum(status_codes == IMPACT_STATUS_TIMEOUT)
        num_nan = np.sum(np.isnan(status_codes))
        print(
            f"Ground: {num_ground}, Building: {num_building}, Timeout: {num_timeout}, NaN Status: {num_nan}")
        final_y_values_bldg = simulation_results_buildings[:, 1]
        valid_y_bldg = final_y_values_bldg[~np.isnan(final_y_values_bldg)]
        if valid_y_bldg.size > 0:
            print(
                f"Min/Max final_y (Buildings): {np.min(valid_y_bldg):.2f} / {np.max(valid_y_bldg):.2f}")
        else:
            print("No valid final_y values found (Buildings).")

        # Visualize impacts WITH buildings
        try:
            plt.figure(figsize=(10, 10))
            extent = [grid_origin[0], grid_origin[0] + GRID_EXTENT_METERS[0],
                      grid_origin[1], grid_origin[1] + GRID_EXTENT_METERS[1]]
            # Plot building grid using imshow
            plt.imshow(building_grid, origin='lower', extent=extent,
                       cmap='gray_r', alpha=0.3, aspect='equal')

            ground = simulation_results_buildings[
                status_codes == IMPACT_STATUS_GROUND]
            building = simulation_results_buildings[
                status_codes == IMPACT_STATUS_BUILDING]
            timeout = simulation_results_buildings[
                status_codes == IMPACT_STATUS_TIMEOUT]
            plt.scatter(ground[:, 0], ground[:, 1], c='blue', s=5,
                        label=f'Ground ({num_ground})', alpha=0.6)
            plt.scatter(building[:, 0], building[:, 1], c='red', s=10,
                        marker='x', label=f'Building ({num_building})',
                        alpha=0.8)
            plt.scatter(timeout[:, 0], timeout[:, 1], c='orange', s=10,
                        marker='s', label=f'Timeout ({num_timeout})', alpha=0.6)
            plt.title("Fragment Impact Locations (With Buildings)")
            plt.xlabel("X Coordinate (m)"), plt.ylabel("Y Coordinate (m)")
            plt.legend(), plt.grid(True, linestyle=':'), plt.axis('equal')
            plt.xlim(extent[0], extent[1]), plt.ylim(extent[2], extent[3])
            plt.show(block=False)
        except Exception as e:
            print(f"Error during 'with buildings' plotting: {e}")
    elif simulation_results_buildings is not None:
        print(
            "\n--- Results WITH Buildings: No particles simulated or all resulted in NaN. ---")
    else:
        print(
            "\n--- Simulation WITH Buildings failed or produced no results. ---")

    # --- Plot WITHOUT Buildings ---
    if simulation_results_no_buildings is not None and \
            simulation_results_no_buildings.shape[0] > 0:
        print(
            f"\n--- Results WITHOUT Buildings ({simulation_results_no_buildings.shape[0]} particles processed) ---")
        status_codes_nb = simulation_results_no_buildings[:, 5]
        num_ground_nb = np.sum(status_codes_nb == IMPACT_STATUS_GROUND)
        num_building_nb = np.sum(status_codes_nb == IMPACT_STATUS_BUILDING)
        num_timeout_nb = np.sum(status_codes_nb == IMPACT_STATUS_TIMEOUT)
        num_nan_nb = np.sum(np.isnan(status_codes_nb))
        print(
            f"Ground: {num_ground_nb}, Building: {num_building_nb}, Timeout: {num_timeout_nb}, NaN Status: {num_nan_nb}")
        final_y_values_nobldg = simulation_results_no_buildings[:, 1]
        valid_y_nobldg = final_y_values_nobldg[~np.isnan(final_y_values_nobldg)]
        if valid_y_nobldg.size > 0:
            print(
                f"Min/Max final_y (No Buildings): {np.min(valid_y_nobldg):.2f} / {np.max(valid_y_nobldg):.2f}")
        else:
            print("No valid final_y values found (No Buildings).")

        # Visualize impacts WITHOUT buildings
        try:
            plt.figure(figsize=(10, 10))
            extent = [grid_origin[0], grid_origin[0] + GRID_EXTENT_METERS[0],
                      grid_origin[1], grid_origin[1] + GRID_EXTENT_METERS[1]]
            ground_nb = simulation_results_no_buildings[
                status_codes_nb == IMPACT_STATUS_GROUND]
            timeout_nb = simulation_results_no_buildings[
                status_codes_nb == IMPACT_STATUS_TIMEOUT]
            plt.scatter(ground_nb[:, 0], ground_nb[:, 1], c='green', s=5,
                        label=f'Ground ({num_ground_nb})', alpha=0.6)
            plt.scatter(timeout_nb[:, 0], timeout_nb[:, 1], c='orange', s=10,
                        marker='s', label=f'Timeout ({num_timeout_nb})',
                        alpha=0.6)
            plt.title("Fragment Impact Locations (NO Buildings)")
            plt.xlabel("X Coordinate (m)"), plt.ylabel("Y Coordinate (m)")
            plt.legend(), plt.grid(True, linestyle=':'), plt.axis('equal')
            plt.xlim(extent[0], extent[1]), plt.ylim(extent[2], extent[3])
            plt.show()  # Use block=True for the last plot
        except Exception as e:
            print(f"Error during 'no buildings' plotting: {e}")
    elif simulation_results_no_buildings is not None:
        print(
            "\n--- Results WITHOUT Buildings: No particles simulated or all resulted in NaN. ---")
    else:
        print(
            "\n--- Simulation WITHOUT Buildings failed or produced no results. ---")
