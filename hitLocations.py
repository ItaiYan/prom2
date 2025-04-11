import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
import time
import multiprocessing as mp
from tqdm import tqdm

class Projectile:
    """
    Represents a projectile subject to gravity and drag.

    Attributes:
        v0 (float): Initial velocity in meters/second.
        theta (float): Launch angle in radians.
        y0 (float): Initial height in meters.
        mass (float): Mass of the projectile in kilograms.
        drag_coef (float): Drag coefficient.
        rho_air (float): Density of air in kg/m^3.
        density (float): Density of the projectile in kg/m^3.
        g (float): Acceleration due to gravity in meters/second^2.
        ks (float): Constant used in cross-sectional area calculation.
        area (float): Cross-sectional area of the projectile in square meters.
        initial_state (numpy.ndarray): Initial state array [x0, y0, vx0, vy0].
    """

    def __init__(self, v0, theta, y0, mass=0.1, drag_coef=0.98,
                 rho_air=1.225, density=7800, ks=0.298):
        """
        Initializes the projectile with the given parameters.
        """
        self.v0 = v0
        self.theta = theta
        self.y0 = y0
        self.mass = mass
        self.drag_coef = drag_coef
        self.density = density
        self.rho_air = rho_air
        self.g = 9.81  # gravitational acceleration
        self.ks = ks   # cross-sectional constant

        # Compute the cross-sectional area from mass and density
        self.area = (self.mass / (self.density * self.ks)) ** (2 / 3)

        # Initial state array: [x-position, y-position, x-velocity, y-velocity]
        self.initial_state = np.array([
            0.0,
            self.y0,
            v0 * np.cos(self.theta),
            v0 * np.sin(self.theta)
        ])

    def derivatives(self, t, state):
        """
        Computes derivatives for the system of ODEs.
        """
        x, y, vx, vy = state
        velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)

        # Calculate drag based on velocity magnitude
        drag_force = 0.5 * self.drag_coef * self.rho_air * self.area * (velocity_magnitude ** 2)

        # Components of drag in x and y directions
        drag_x = -drag_force * vx / velocity_magnitude if velocity_magnitude > 0 else 0
        drag_y = -drag_force * vy / velocity_magnitude if velocity_magnitude > 0 else 0

        # Accelerations in x and y
        ax = drag_x / self.mass
        ay = -self.g + (drag_y / self.mass)

        return np.array([vx, vy, ax, ay])

    def simulate(self, t_max=10000.0, rtol=1e-12, atol=1e-12):
        """
        Simulates projectile motion up to t_max using SciPy's RK45.
        """
        # Initialize the RK45 solver with the projectile's derivatives
        solver = RK45(
            fun=self.derivatives,
            t0=0.0,
            y0=self.initial_state,
            t_bound=t_max,
            rtol=rtol,
            atol=atol,
            vectorized=False
        )

        times = [solver.t]     # store time steps
        states = [solver.y]    # store state vectors at each step

        # Continue stepping while the projectile is above ground level
        while solver.y[1] > 0 and solver.status == "running":
            solver.step()
            times.append(solver.t)
            states.append(solver.y)

        return np.array(times), np.array(states)


def run_simulation(initial_conditions):
    """
    Runs a single projectile simulation with given initial conditions.
    """
    if len(initial_conditions) == 4:
        # Unpack initial conditions
        v0 = initial_conditions[0]
        theta = initial_conditions[1]
        y0 = 0.001  # small offset above ground
        mass = initial_conditions[2]
        density = initial_conditions[3]
    elif len(initial_conditions) == 3:
        v0 = initial_conditions[0]
        theta = initial_conditions[1]
        y0 = 0.001  # small offset above ground
        mass = initial_conditions[2]
        density = 7800  # default density

    # Create a Projectile instance and simulate
    projectile = Projectile(v0, theta, y0, mass=mass, density=density)
    times, states = projectile.simulate()

    x_positions = states[:, 0]
    vx_values = states[:, 2]
    vy_values = states[:, 3]

    # Final horizontal position
    final_x = x_positions[-1]
    # Final speed
    final_speed = np.sqrt(vx_values[-1] ** 2 + vy_values[-1] ** 2)
    # Final angle (absolute value inside arctan)
    final_angle = np.arctan(abs(vx_values[-1] / vy_values[-1]))

    return np.array([final_x, final_speed, final_angle, mass])


def main(initial_conditions):
    """
    Entry point to handle multiple projectile simulations in parallel
    and aggregate the results.
    """
    # Select relevant columns [v0, theta, mass, density]
    selected_cols = initial_conditions[:, [0, 1, 3, 4]]

    # Run multiple RK45 simulations in parallel
    result = run_parallel_rk45(selected_cols)

    # Initialize tracking variables
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    shreds = []

    # Convert polar to Cartesian using phi stored in initial_conditions
    for i in range(len(result)):
        # Calculate final x,y based on the angle phi
        current_x = np.cos(initial_conditions[i][2]) * result[i][0]
        current_y = np.sin(initial_conditions[i][2]) * result[i][0]

        # Data structure: (x_position, y_position, speed, angle, mass)
        shreds.append((current_x, current_y, result[i][1], result[i][2], result[i][3]))

        # Track min/max
        max_x = max(max_x, current_x)
        min_x = min(min_x, current_x)
        max_y = max(max_y, current_y)
        min_y = min(min_y, current_y)

    return np.array(shreds)


def run_parallel_rk45(initial_conditions):
    """
    Distributes multiple simulations across available CPU cores.
    """
    print("strting")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(run_simulation, initial_conditions),
            total=len(initial_conditions)
        ))
    return results


if __name__ == "__main__":
    # Example for large number of simulations
    print("started")
    # v0, thaeta, phi, mass, density
    initial_conds = [[1334, np.radians(i), 0, 55.8, 7800] for i in range(45,
                                                                          50)]
    initial_conds = numpy.array(initial_conds)

    print("started")
    t = time.time()
    # returns: (x_position, y_position, speed, angle, mass)
    print (main(initial_conds))
    e = time.time()
