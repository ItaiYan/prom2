import numpy as np
import pandas as pd
import json
from dataclasses import dataclass

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
    return (np.log(U) ** 2) * mean_mass


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

    bomb = Bomb(
        num_of_fragments=num_of_fragments,
        mean_mass=mean_mass,
        velocities=mean_velocity,
        density=7800,
        num_of_sections=len(num_of_fragments)
    )
    theta_hit = data["angle"]
    velocity = data["velocity"]
    return bomb, theta_hit, velocity


def test():
    bomb, theta_hit, velocity = open_data()
    fragments = generate_fragments(bomb)
    transformed = transform_coordinates(fragments, velocity,
                                        np.radians(theta_hit))
    return transformed
