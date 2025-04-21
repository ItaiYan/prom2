import numpy as np
import pandas as pd
import json
from dataclasses import dataclass
import math

# Constants for readability
V, THETA, PHI, MASS, DENSITY = range(5)


@dataclass
class Bomb:
    num_of_fragments: np.ndarray
    mean_mass: np.ndarray
    velocities: np.ndarray
    density: float
    num_of_sections: int


def generate_mass_vectorized(mean_mass: float, size: int):
    U = np.random.uniform(0, 1, size)
    return (np.log(U) ** 2) * (mean_mass / 2)





def generate_fragments(bomb: Bomb) -> np.ndarray:
    fragments = []
    for section in range(bomb.num_of_sections):
        section_fragments = []
        mass_accum = 0
        mass_target = 2 * bomb.mean_mass[section] * bomb.num_of_fragments[
            section]

        dphi = np.pi / bomb.num_of_sections
        theta_min = section * dphi
        theta_max = (section + 1) * dphi
        v = bomb.velocities[section]
        d = bomb.density
        mean_mass = bomb.mean_mass[section]

        while mass_accum < mass_target:
            batch_size = 500  # process in chunks to reduce overhead
            theta = np.random.uniform(theta_min, theta_max, batch_size)
            phi = np.random.uniform(0, np.pi, batch_size)
            mass = generate_mass_vectorized(mean_mass, batch_size)
            velocity = np.full(batch_size, v)
            density = np.full(batch_size, d)

            new_fragments = np.stack([velocity, theta, phi, mass, density],
                                     axis=-1)

            total_mass = np.cumsum(mass)
            stop_idx = np.searchsorted(mass_accum + total_mass, mass_target)

            if stop_idx < batch_size:
                section_fragments.append(new_fragments[:stop_idx + 1])
                mass_accum += total_mass[stop_idx]
                break
            else:
                section_fragments.append(new_fragments)
                mass_accum += total_mass[-1]

        section_all = np.concatenate(section_fragments, axis=0)
        section_all = section_all[(section_all[:, 3] >= 1e-3) & (section_all[:, 3] <= 2)]
        fragments.append(section_all)

    return np.concatenate(fragments, axis=0)


def transform_coordinates(fragments: np.ndarray, v_impact,
                          theta_impact) -> np.ndarray:
    v = fragments[:, V]
    theta = fragments[:, THETA]
    phi = fragments[:, PHI]

    v_x = v * np.sin(theta) * np.cos(phi)
    v_y = v * np.sin(theta) * np.sin(phi)
    v_z = v * np.cos(theta)

    # Rotate to rocket frame
    v_x_initial = np.cos(theta_impact) * v_z + np.sin(theta_impact) * v_x
    v_y_initial = v_y
    v_z_initial = -v_z * np.sin(theta_impact) + v_x * np.cos(theta_impact)

    v_cart = np.stack([v_x_initial, v_y_initial, v_z_initial], axis=-1)
    rocket_v = np.array([
        v_impact * np.cos(theta_impact),
        0,
        -v_impact * np.sin(theta_impact)
    ])
    total_v = v_cart + rocket_v

    v_tot = np.linalg.norm(total_v, axis=1)
    theta_new = np.arctan2(total_v[:, 2],
                           np.sqrt(total_v[:, 0] ** 2 + total_v[:, 1] ** 2))
    phi_new = np.arctan2(total_v[:, 1], total_v[:, 0])

    mass = fragments[:, MASS]
    density = fragments[:, DENSITY]

    return np.stack([v_tot, theta_new, phi_new, mass, density], axis=-1)


def open_data():
    with open("input.JSON") as f:
        data = json.load(f)
    bomb_type = data["bomb type"]
    bomb_data = pd.read_excel(f"{bomb_type}_frag.xlsx")

    num_of_fragments = bomb_data["N"].to_numpy()
    mean_mass = bomb_data["mean weight (grain)"].to_numpy() * 0.00006479891
    mean_velocity = bomb_data["mean velocity (ft/s)"].to_numpy() * 0.3048
    urban_area = bool(data["urban area"])

    bomb = Bomb(
        num_of_fragments=num_of_fragments,
        mean_mass=mean_mass,
        velocities=mean_velocity,
        density=7800,
        num_of_sections=len(num_of_fragments)
    )
    theta_hit = data["angle"]
    velocity = data["velocity"]
    return bomb, theta_hit, velocity, urban_area

def reduce_speed_all_fragments(fragments: np.ndarray, h=0.02):
    # Constants for Mild Steel from THOR formula table
    c11 = 4.356
    c12 = 0.674
    c13 = -0.791
    c14 = 0.989
    c15 = 0.434
    c31 = -1.195
    c32 = 0.234
    c33 = 0.743
    c34 = 0.469
    c35 = 0.483

    V_R = fragments[:, V]
    m0 = fragments[:, MASS]
    density = fragments[:, DENSITY]
    theta = fragments[:, THETA]

    A = (m0 / (density * 0.298)) ** (2 / 3)

    # First reduction (Velocity)
    num1 = 0.3048e11 * (61023.75 * h) ** c12
    den1 = (15432.1 * m0) ** c13 * (1 / np.cos(theta)) ** c14 * (3.28084 * V_R) ** c15
    fragments[:, V] = V_R - num1 / den1

    # Second reduction (Mass)
    num2 = 6.48e26 * (61023.75 * h) ** c32
    den2 = (15432.1 * m0) ** c33 * (1 / np.cos(theta)) ** c34 * (3.28084 * V_R) ** c35
    fragments[:, MASS] = m0 - num2 / den2

    return fragments


def test():
    bomb, theta_hit, velocity, urban_area = open_data()
    fragments = generate_fragments(bomb)
    transformed = transform_coordinates(fragments, velocity,
                                        np.radians(theta_hit))
    if urban_area:
        transformed = reduce_speed_all_fragments(transformed)

    return transformed
