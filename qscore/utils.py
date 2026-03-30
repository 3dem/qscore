from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from qscore.mrc_utils import MRCObject


def sample_uniformly_on_sphere(sphere_radius: float, num_points: int) -> np.ndarray:
    u = np.random.random(num_points)
    v = np.random.random(num_points)
    z = 2.0 * u - 1.0
    phi = 2.0 * np.pi * v
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.empty((num_points, 3), dtype=np.float64)
    points[:, 0] = r_xy * np.cos(phi)
    points[:, 1] = r_xy * np.sin(phi)
    points[:, 2] = z
    points *= sphere_radius
    return points

def get_reference_gaussian_params(map: MRCObject) -> Tuple[float, float]:
    map_max = np.max(map.grid)
    map_min = np.min(map.grid)
    map_mean = np.mean(map.grid)
    map_std = np.std(map.grid)
    high_value = min(map_mean + 10 * map_std, map_max)
    low_value = max(map_mean - map_std, map_min)
    reference_gaussian_height = high_value - low_value
    reference_gaussian_offset = low_value
    return reference_gaussian_height, reference_gaussian_offset


def get_radial_points(
        atoms: np.ndarray,
        sphere_radius: float,
        num_points: int,
        kdtree: Optional[cKDTree] = None,
        query_workers: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    num_atoms = len(atoms)
    radial_points = np.zeros((num_atoms, num_points, 3), dtype=atoms.dtype)
    if kdtree is None:
        kdtree = cKDTree(atoms)
    fill_counts = np.zeros(num_atoms, dtype=np.int16)
    atom_indices = np.arange(num_atoms)
    for _ in range(100):
        atoms_left = fill_counts < num_points
        if not np.any(atoms_left):
            break
        atom_indices_left = atom_indices[atoms_left]
        num_atoms_left = len(atom_indices_left)
        sphere_points = sample_uniformly_on_sphere(
            sphere_radius, num_points * num_atoms_left
        ).reshape((num_atoms_left, num_points, 3)).astype(atoms.dtype, copy=False)
        sphere_points += atoms[atoms_left, None]
        indices = kdtree.query(sphere_points, k=1, workers=query_workers)[1]
        valid = indices == atom_indices_left[:, None]
        rank = np.cumsum(valid, axis=1) - 1
        remaining = (num_points - fill_counts[atom_indices_left])[:, None]
        chosen = valid & (rank < remaining)
        if not np.any(chosen):
            continue
        atom_assign = np.broadcast_to(atom_indices_left[:, None], chosen.shape)[chosen]
        slot_assign = (fill_counts[atom_indices_left][:, None] + rank)[chosen]
        radial_points[atom_assign, slot_assign] = sphere_points[chosen]
        fill_counts[atom_indices_left] += chosen.sum(axis=1).astype(fill_counts.dtype)
    point_exists = np.arange(num_points)[None, :] < fill_counts[:, None]
    return radial_points, point_exists


def interpolate_grid_at_points(points: np.ndarray, map: MRCObject) -> np.ndarray:
    points = np.flip(points, axis=-1)
    # Origin should be flipped the same way, thank you Sjors
    flipped_origin = np.flip(map.global_origin, axis=-1)
    p = ((points - flipped_origin[None]) / map.voxel_size).reshape(-1, 3).T
    values = map_coordinates(
        map.grid,
        p,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return values.reshape(points.shape[:-1])
