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
        mass_target = bomb.mean_mass[section] * bomb.num_of_fragments[
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
            phi = np.random.uniform(0, 2 * np.pi, batch_size)
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
        #res = section_all[section_all[:, 3] > 0.02]
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
    # Constants for Mild Steel (from your table)
    c11, c12, c13, c14, c15 = 1.999, 0.499, -0.502, 0.655, 0.818
    c31, c32, c33, c34, c35 = -1.856, 0.506, 0.350, 0.777, 0.934
    ks = 0.298

    # Indices (make sure they match your fragments array structure)

    V_R = fragments[:, V]
    mass = fragments[:, MASS]
    theta = fragments[:, THETA]
    density = fragments[:, DENSITY]
    # Correct power operator for area ∝ (mass / density/ks)^(2/3)
    areas = (mass / (density * ks)) ** (2 / 3)

    # Safe values
    safe_m0 = np.clip(mass, 1e-8, None)  # mass must be positive
    safe_VR = np.clip(V_R, 1e-8, None)  # velocity must be positive
    cos_theta = np.clip(np.cos(theta), 1e-6, 1.0)
    sec_theta = 1 / cos_theta  # secant(theta)

    # Calculate new Residual Velocity Vr directly
    num_vr = 0.3048 * (10 ** c11) * (61023.75 * h * areas) ** c12
    den_vr = (15432.1 * safe_m0) ** c13 * (sec_theta) ** c14 * (3.28084 * safe_VR) ** c15
    Vr = safe_VR - num_vr * den_vr
    mask_v = Vr > 0.001

    # Calculate new Residual Mass mr directly
    num_mr = 6.48 * (10 ** (c31 - 5)) * (61023.75 * h * areas) ** c32
    den_mr = (15432.1 * safe_m0) ** c33 * (sec_theta) ** c34 * (3.28084 * safe_VR) ** c35
    mr = safe_m0 - num_mr * den_mr
    mask_m = mr > 0.001

    # Update the fragments
    fragments[:, V] = Vr
    fragments[:, MASS] = mr


    full_mask = mask_m & mask_v
    return fragments[full_mask]


def test():
    bomb, theta_hit, velocity, urban_area = open_data()
    fragments = generate_fragments(bomb)
    transformed = transform_coordinates(fragments, velocity,
                                        np.radians(theta_hit))


    if urban_area:
        transformed = reduce_speed_all_fragments(transformed)

    return transformed



# bomb, theta_hit, velocity, urban_area = open_data()
# fragments = generate_fragments(bomb)
# transformed = transform_coordinates(fragments, velocity,
#                                         np.radians(theta_hit))
# if urban_area:
#      transformed = reduce_speed_all_fragments(transformed)