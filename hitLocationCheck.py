import numpy as np
import particleGenerator

V, THETA, PHI, MASS, DENSITY = range(5)
import numpy as np

import numpy as np

import numpy as np

def simulate_particles_rk4(
        fragments, d_c=1.1, air_density=1.225, dt=0.01, t_max=10.0, g=9.81
):
    """
    Vectorized RK4 integrator for particles with drag in a spherical coordinate system.

    Args:
        fragments: (N, 5) array with columns [mass, density, total_velocity, theta, phi]
        d_c: drag coefficient (scalar or array of shape (N,))
        air_density: air density (kg/m^3)
        dt: initial time step (s)
        t_max: max simulation time (s)
        g: gravity constant (m/s^2)

    Returns:
        impact_positions: (N, 3) array of [x, y, z] positions where each particle hits z=0
    """
    # Define column indices
    N = len(fragments)

    # Validate inputs
    if not np.all(np.isfinite(fragments)):
        raise ValueError("Fragments contains non-finite values")
    if np.any(fragments[:, MASS] <= 0):
        raise ValueError("Masses must be positive")
    if np.any(fragments[:, DENSITY] <= 0):
        raise ValueError("Densities must be positive")
    if np.any(fragments[:, V] < 0):
        raise ValueError("Total velocities must be non-negative")

    # Extract initial conditions
    masses = fragments[:, MASS]  # (N,)
    densities = fragments[:, DENSITY]  # (N,)
    total_v = fragments[:, V]  # (N,)
    theta = fragments[:, THETA]  # (N,)
    phi = fragments[:, PHI]  # (N,)

    # Precompute constants
    vr = total_v * np.cos(theta)  # (N,)
    vz = total_v * np.sin(theta)  # (N,)
    velocities = np.column_stack((vr, vz))  # (N, 2)
    positions = np.zeros((N, 2))  # (N, 2) [x, z]
    areas = (masses / (densities * (4/3) * np.pi)) ** (2/3)  # (N,)
    drag_coeffs = d_c * np.ones(N) if np.isscalar(d_c) else d_c  # (N,)
    drag_factor = -0.5 * air_density * drag_coeffs / masses  # (N,)
    gravity = np.array([0, -g])  # (1, 2)

    # Initialize state vector [x, z, vx, vz]
    state_vector = np.column_stack((positions, velocities))  # (N, 4)
    impact_positions = np.full((N, 3), np.nan)  # (N, 3) [x, y, z]
    active = np.ones(N, dtype=bool)

    def xyz_locations(state: np.ndarray, phi: np.ndarray):
        r = state[:, 0]  # (N,)
        z = state[:, 1]  # (N,)
        x = r * np.cos(phi)  # (N,)
        y = r * np.sin(phi)  # (N,)
        return np.column_stack((x, y, z))  # (N, 3)

    def acceleration(v: np.ndarray):
        v_mag = np.sqrt(np.sum(v**2, axis=1, keepdims=True))  # (N, 1)
        v_mag_safe = np.where(v_mag > 1e-10, v_mag, 1e-10)  # (N, 1)
        drag = (drag_factor[:, None] * areas[:, None] * v_mag_safe) * v  # (N, 2)
        return drag + gravity  # (N, 2)

    def dsdt(state: np.ndarray):
        v = state[:, 2:4]  # (N, 2)
        a = acceleration(v)  # (N, 2)
        return np.column_stack((v, a))  # (N, 4)

    t = 0.0
    while t < t_max and np.any(active):
        # Adaptive time step (optional)
        v_mag = np.sqrt(np.sum(state_vector[:, 2:4]**2, axis=1))  # (N,)
        # dt_adaptive = min(dt, 0.01 / (np.max(v_mag) + 1e-10)) if np.any(v_mag > 0) else dt
        dt_adaptive = 0.1
        z_prev = state_vector[active, 1]  # (M,)
        # RK4 integration
        k1 = dsdt(state_vector)
        k2 = dsdt(state_vector + 0.5 * dt_adaptive * k1)
        k3 = dsdt(state_vector + 0.5 * dt_adaptive * k2)
        k4 = dsdt(state_vector + dt_adaptive * k3)
        state_vector = state_vector + (dt_adaptive / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Check for ground impact

        z_curr = state_vector[active, 1]  # (M,)
        crossed = (z_prev > 0) & (z_curr <= 0)
        crossed_full = np.zeros(N, dtype=bool)
        crossed_full[active] = crossed

        if np.any(crossed_full):
            active_indices = np.where(active)[0]
            crossed_indices = active_indices[crossed]
            alpha = z_prev[crossed] / (z_prev[crossed] - z_curr[crossed])
            alpha = np.clip(alpha, 0, 1)
            state_interp = (state_vector[crossed_indices] * alpha[:, None] +
                          state_vector[crossed_indices] * (1 - alpha[:, None]))
            impact_positions[crossed_indices] = xyz_locations(state_interp, phi[crossed_indices])
            active[crossed_indices] = False

        t += dt_adaptive

    return impact_positions
# N = 10000
# angles = np.radians(np.random.uniform(15, 75, N))
# speeds = np.random.uniform(100, 300, N)
#
# vx = speeds * np.cos(angles)
# vy = speeds * np.sin(angles)
#
# positions = np.zeros((N, 2))  # start at origin
# velocities = np.stack([vx, vy], axis=1)
# masses = np.full(N, 0.05)  # 50g per particle
# areas = np.full(N, 0.0002)  # m^2
# drag_coeffs = np.full(N, 0.47)  # sphere

fragments = particleGenerator.test()
fragments = fragments[: 10]

impact_xyz = simulate_particles_rk4(fragments)
print(impact_xyz)

# import matplotlib.pyplot as plt
# import matplotlib

