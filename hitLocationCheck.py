import numpy as np
import numba # Import Numba
import time

# Assuming particleGenerator.py exists and has a test() function
# that returns fragments like: [V, theta, phi, mass, density]
import particleGenerator

# --- Derivative Function (Fixed and JIT-compiled) ---
@numba.jit(nopython=True, cache=True)
def projectile_deriv_numba(state, masses, areas, t):
    rho = 1.225  # Air density
    C_D = 1.1   # Drag coefficient (example)
    vx = state[:, 2]
    vy = state[:, 3]
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    safe_velocity_magnitude = velocity_magnitude + 1e-9 # Avoid division by zero

    # Calculate magnitude of drag force
    drag_force_magnitude = 0.5 * rho * C_D * areas * velocity_magnitude**2

    # Calculate components of drag force (opposite to velocity direction)
    drag_x = -drag_force_magnitude * (vx / safe_velocity_magnitude)
    drag_y = -drag_force_magnitude * (vy / safe_velocity_magnitude)

    # Calculate acceleration components
    ax = drag_x / masses
    ay = -9.81 + (drag_y / masses) # Gravity + drag

    return np.column_stack((vx, vy, ax, ay))


# --- Core Simulation Function (Fixed and JIT-compiled) ---
@numba.jit(nopython=True, cache=True)
def rk4_vectorized_simulation_numba(initial_state_and_mass, dt, t_max):
    """
    Performs a vectorized RK4 simulation for N particles until they hit the ground.
    Optimized with Numba.

    Args:
        initial_state_and_mass (np.ndarray): Shape (N, 5). Columns MUST be
                                             [x, y, vx, vy, mass].
        deriv_func (callable): A Numba-compiled function that computes derivatives.
                               Signature: deriv_func(state_active, masses_active, t) -> derivatives
                               Input state shape: (M, 4) [x, y, vx, vy].
                               Input masses shape: (M,).
                               Output derivatives shape: (M, 4) [vx, vy, ax, ay].
        dt (float): The time step for the simulation.
        t_max (float): The maximum simulation time.

    Returns:
        np.ndarray: Shape (N,). Contains the x-position of each particle when
                    it hits the ground (y <= 0). Values are np.nan if the
                    particle doesn't hit the ground within t_max.
    """
    n_particles = initial_state_and_mass.shape[0]

    # --- Initialization ---
    # Separate state [x, y, vx, vy] and mass
    state = initial_state_and_mass[:, :4].copy().astype(np.float64)
    masses = initial_state_and_mass[:, 4].copy().astype(np.float64) # Extract mass column
    areas = initial_state_and_mass[:, 7].copy().astype(np.float64) # Extract
    # area column
    t = 0.0
    # Array to store the final x-position upon ground impact
    final_x = np.full(n_particles, np.nan, dtype=np.float64) # Specify dtype for Numba
    final_v_x = np.full(n_particles, np.nan, dtype=np.float64) # Specify dtype for Numba
    final_v_y = np.full(n_particles, np.nan, dtype=np.float64) # Specify dtype for Numba

    # Boolean mask to track which particles are still active
    active_mask = np.ones(n_particles, dtype=np.bool_)

    # --- Check for particles starting at or below ground ---
    for i in range(n_particles):
        # state[:, 1] is the y-coordinate
        if state[i, 1] <= 0:
            final_x[i] = state[i, 0] # state[:, 0] is the x-coordinate
            final_v_x[i] = state[i, 2]
            final_v_y[i] = state[i, 3]
            active_mask[i] = False

    # --- Simulation Loop ---
    while t < t_max:
        # Find indices of active particles
        current_active_indices = np.where(active_mask)[0]
        num_active = len(current_active_indices)

        if num_active == 0:
            print("All particles have hit the ground.")
            break # Exit if no particles are active

        # Get the state and mass of currently active particles
        state_active = state[current_active_indices]
        masses_active = masses[current_active_indices] # Get masses for active particles
        state_prev_active = state_active.copy() # Store previous state for interpolation
        areas_active = areas[current_active_indices] # Get areas for active particles
        # --- RK4 Step (Applied only to active particles) ---
        # Pass active state AND active masses to deriv_func
        k1_active = projectile_deriv_numba(state_active, masses_active, areas_active, t) * dt

        state_temp_active = state_active + 0.5 * k1_active
        # Need to recalculate derivatives at the intermediate step
        k2_active = projectile_deriv_numba(state_temp_active, masses_active, areas_active,
                               t + 0.5 * dt) * dt

        state_temp_active = state_active + 0.5 * k2_active
        k3_active = projectile_deriv_numba(state_temp_active, masses_active, areas_active,  (t +
                                                                        0.5 * dt)) * dt

        state_temp_active = state_active + k3_active
        k4_active = projectile_deriv_numba(state_temp_active, masses_active, areas_active,
                               t + dt) * dt

        # Update active states
        state_active += (k1_active + 2.0*k2_active + 2.0*k3_active +
                         k4_active) / 6.0
        t += dt


        # Update the main state array using the active indices
        state[current_active_indices] = state_active

        # --- Check for Ground Collision (only among active particles) ---
        for i in range(num_active):
            idx_global = current_active_indices[i]
            # Check if particle *was* active and now its y (index 1) is <= 0
            if active_mask[idx_global] and state_active[i, 1] <= 0:
                # Get previous and current states for this specific particle
                # Use correct indices: 0 for x, 1 for y
                y_prev = state_prev_active[i, 1]
                x_prev = state_prev_active[i, 0]
                y_curr = state_active[i, 1]
                x_curr = state_active[i, 0]

                # --- Interpolation for Impact Point ---
                x_impact = x_curr # Default to current x
                v_x_impact = state_active[i, 2]
                v_y_impact = state_active[i, 3]
                if y_prev > 0: # Only interpolate if it crossed the boundary
                    delta_y = y_prev - y_curr
                    # Avoid division by zero or very small numbers
                    if delta_y > 1e-12:
                        alpha = y_prev / delta_y
                        # Clamp alpha for safety (should be between 0 and 1)
                        alpha = max(0.0, min(1.0, alpha))
                        x_impact = x_prev + alpha * (x_curr - x_prev)
                        v_x_impact = state_prev_active[i, 2] + alpha * (state_active[i, 2] - state_prev_active[i, 2])
                        v_y_impact = state_prev_active[i, 3] + alpha * (state_active[i, 3] - state_prev_active[i, 3])
                    else:
                        # If delta_y is tiny, it likely just touched or slid
                        # Use previous x if it was above ground, else current
                        x_impact = x_prev
                        v_x_impact = state_prev_active[i, 2]
                        v_y_impact = state_prev_active[i, 3]


                # Store the impact x-position
                final_x[idx_global] = x_impact
                final_v_x[idx_global] = v_x_impact
                final_v_y[idx_global] = v_y_impact

                # Deactivate collided particle
                active_mask[idx_global] = False

    return np.column_stack((final_x, final_v_x, final_v_y))


# --- Main Runner Function (Fixed) ---
def run_particle_simulation(initial_conditions_full, dt=0.01, t_max=20.0):
    """
    Sets up and runs the Numba-optimized particle simulation.

    Args:
        initial_conditions_full (np.ndarray): Shape (N, 5 or more).
                                              Columns must start with [x, y, vx, vy, mass, ...].
        dt (float, optional): Time step for the simulation. Defaults to 0.01.
        t_max (float, optional): Maximum simulation time. Defaults to 20.0.

    Returns:
        np.ndarray: Shape (N,). Contains the x-position of each particle when
                    it hits the ground (y <= 0). Values are np.nan if the
                    particle doesn't hit the ground within t_max.
    """
    print(f"Starting simulation for {initial_conditions_full.shape[0]} particles...")
    start_time = time.time()

    # --- Prepare input for Numba function: Need [x, y, vx, vy, mass] ---
    # if initial_conditions_full.shape[1] < 5:
    #     raise ValueError("initial_conditions_full must have at least 5 columns (x, y, vx, vy, mass)")
    # Select the first 5 columns required by the Numba function
    # initial_state_and_mass = initial_conditions_full[:, :5]

    # --- Run Core Simulation ---
    # Pass the Numba-compiled derivative function directly
    # impact_state: N x [x, vx, vy]
    impact_state = rk4_vectorized_simulation_numba(
        initial_conditions_full,  # Pass the array including mass
        dt,
        t_max
    )

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.4f} seconds.")
     # Return x-positions and
    # y-positions
    return impact_state

# --- Particle Data Conversion (Assumed Correct) ---
def particle_conversion_for_simulation(fragments, ks=0.298, default_C_D=1.1):
    """
    :param fragments: array [V, theta, phi, mass, density]
    :return: array [x, y, vx, vy, mass, phi, density, area, C_D]
    """
    # index names
    v_idx, theta_idx, phi_idx, mass_idx, rho_idx = range(5)

    N = fragments.shape[0]
    x = np.zeros(N)
    y = np.ones(N) * 1

    vx = fragments[:, v_idx] * np.cos(fragments[:, theta_idx])
    vy = fragments[:, v_idx] * np.sin(fragments[:, theta_idx])

    mass = fragments[:, mass_idx]
    density = fragments[:, rho_idx]

    # Correct power operator for area ∝ (mass / density/ks)^(2/3)
    areas = (mass / (density * ks)) ** (2/3)

    # uniform drag coefficient if you don’t vary it per particle
    cds = np.ones(N) * default_C_D

    return np.column_stack((x, y, vx, vy, mass, fragments[:, phi_idx],
                            density, areas, cds))


def main(fragments: np.ndarray):
    """
    Main function to run the simulation.
    """
    # Convert fragments to the initial conditions format needed
    # [x, y, vx, vy, mass, phi, density]
    initial_conditions = particle_conversion_for_simulation(fragments)
    impact_state = run_particle_simulation(initial_conditions)

    #convert the impact state and the initial conditions to output format: [x, y, v, theta_impact, mass]
    impact_x = impact_state[:, 0] * np.cos(initial_conditions[:, 5])
    impact_y = impact_state[:, 0] * np.sin(initial_conditions[:, 5])
    theta_impact = -np.arctan2(impact_state[:, 2], impact_state[:, 1])
    v_impact = np.sqrt(impact_state[:, 1]**2 + impact_state[:, 2]**2)
    return np.column_stack((impact_x, impact_y, v_impact, theta_impact,
                            initial_conditions[:, 4]))
# --- Example Usage Block ---
if __name__ == "__main__":
    print("--- Running Simulation Example ---")

    # Generate some test particle data using your generator
    # Assuming particleGenerator.test() returns N x [V, theta, phi, mass, density]
    try:
        fragments = particleGenerator.test()
        print(f"Generated {fragments.shape[0]} fragments.")
    except AttributeError:
        print("Error: 'particleGenerator' module or 'test' function not found.")
        print("Using placeholder data instead.")
        # Placeholder if particleGenerator is not available
        N_PARTICLES_EXAMPLE = 10000
        fragments = np.zeros((N_PARTICLES_EXAMPLE, 5))
        fragments[:, 0] = np.random.uniform(10, 100, N_PARTICLES_EXAMPLE) # V
        fragments[:, 1] = np.random.uniform(0.1, np.pi/2 - 0.1, N_PARTICLES_EXAMPLE) # theta (avoid exactly 0 or pi/2)
        fragments[:, 2] = np.random.uniform(0, 2*np.pi, N_PARTICLES_EXAMPLE) # phi (unused in this 2D sim)
        fragments[:, 3] = np.random.uniform(0.1, 5.0, N_PARTICLES_EXAMPLE) # mass
        fragments[:, 4] = np.random.uniform(1000, 8000, N_PARTICLES_EXAMPLE) # density (unused in this sim)


    # Convert fragments to the initial conditions format needed
    # [x, y, vx, vy, mass, phi, density]
    initial_conditions = particle_conversion_for_simulation(fragments)
    N_PARTICLES = initial_conditions.shape[0]


    # --- Run Simulation using the main function ---
    # First run might be slower due to Numba compilation
    impact_x = run_particle_simulation(initial_conditions, dt=0.1,
                                       t_max=300.0)

    # --- Output Results ---
    num_hit_ground = np.sum(~np.isnan(impact_x))
    num_not_hit = np.sum(np.isnan(impact_x))
    print(f"\n--- Simulation Results ({N_PARTICLES} particles) ---")
    print(f"Particles that hit ground: {num_hit_ground}")
    print(f"Particles that did not hit ground within t_max: {num_not_hit}")
    if impact_x is not None:
        print(f"Shape of output: {impact_x.shape}")

        # Print results for the first few particles
        print("\nImpact results for first 10 particles:")
        for i in range(min(N_PARTICLES, 10000000)):
            # Access original initial conditions for printing comparison
            # Indices: 2=vx, 3=vy
            print(f"Particle {i}: Initial (vx,vy)=({initial_conditions[i, 2]:.2f}, {initial_conditions[i, 3]:.2f}) -> Final x = {impact_x[i]:.2f}")

        # Example: Find average impact distance for those that hit the ground
        if num_hit_ground > 0:
            valid_impacts = impact_x[~np.isnan(impact_x)]
            print(f"\nAverage impact distance (for those that hit): {np.mean(valid_impacts):.2f}")
        else:
            print("\nNo particles hit the ground within the simulation time.")

    # --- Optional: Run again to see cached performance ---
    print("\n--- Running Simulation Example Again (should use Numba cache) ---")
    impact_x_cached = run_particle_simulation(initial_conditions, dt=0.01,
                                              t_max=30.0)
    # (Optionally print results again or just show timing difference)


