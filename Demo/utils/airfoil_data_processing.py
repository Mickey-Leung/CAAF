"""Airfoil interpolation and per-case scaling with aligned timestamps and grids."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# Interpolate each surface onto the reference grid and apply the case weighting.
def load_airfoil_data(data_dir: Path, samples_per_case=30000):
    """Return pressure, lift, and wall coordinates for the three training cases."""
    data_dir = Path(data_dir)
    if isinstance(samples_per_case, (bool, np.bool_)) or not isinstance(samples_per_case, (int, np.integer)) or samples_per_case < 2:
        raise ValueError("samples_per_case must be an integer >= 2.")
    cases = ("airfoil_5", "airfoil_11", "cylinder_5")
    augmentation = (0, 0, 2)
    wall1 = np.genfromtxt(data_dir / "cylinder_5_xy_sort_wall1.dat", delimiter=",")
    n_grid = int(wall1[0, 0])
    wall1 = wall1[1:n_grid + 1, :2]
    wall2 = np.loadtxt(data_dir / "cylinder_5_xy_sort_wall2.dat", skiprows=1)[:n_grid, :2]
    pressures, lifts = [], []
    for case, extra_copies in zip(cases, augmentation):
        # These CSV files have no header; retain the first sample for time alignment.
        lift_data = pd.read_csv(data_dir / f"{case}_CL.csv", header=None).to_numpy()
        raw1 = np.loadtxt(data_dir / f"{case}_wall1_surfacepressure_span.dat",
                          delimiter=",", usecols=range(n_grid + 1))
        raw2 = np.loadtxt(data_dir / f"{case}_wall2_surfacepressure_span.dat",
                          delimiter=",", usecols=range(n_grid + 1))
        pressure_times = lift_data[np.arange(3, len(raw1) * 4, 4), 0]
        uniform_times = np.arange(pressure_times[0], pressure_times[-1], 0.0025)
        if len(uniform_times) < samples_per_case:
            raise ValueError(f"{case} has fewer than {samples_per_case} interpolated samples.")
        times = uniform_times[:samples_per_case]
        pressure1 = interp1d(pressure_times, raw1[:, 1:], kind="cubic", axis=0)(times)
        pressure2 = interp1d(pressure_times, raw2[:, 1:], kind="cubic", axis=0)(times)
        lift = interp1d(lift_data[:, 0], lift_data[:, 1], kind="cubic",
                       fill_value="extrapolate")(times)

        # Each surface uses its own source coordinates before spatial interpolation.
        source_grid = np.genfromtxt(data_dir / f"{case}_xy_sort_wall1.dat", delimiter=",")
        source_x = source_grid[1:int(source_grid[0, 0]) + 1, 0]
        source_x2 = np.loadtxt(data_dir / f"{case}_xy_sort_wall2.dat", skiprows=1)[:n_grid, 0]
        pressure1 = interp1d(source_x, pressure1, kind="cubic", axis=1,
                            fill_value="extrapolate")(wall1[:, 0])
        pressure2 = interp1d(source_x2, pressure2, kind="cubic", axis=1,
                            fill_value="extrapolate")(wall2[:, 0])
        pressure = np.hstack((pressure1, pressure2))
        pressure_scale = np.ptp(pressure, axis=0)
        lift_scale = np.ptp(lift)
        if not np.isfinite(pressure).all() or not np.isfinite(lift).all() or lift_scale == 0:
            raise ValueError(f"{case} must contain finite pressure and varying finite lift values.")
        pressure = (pressure - pressure.mean(axis=0)) / np.where(pressure_scale == 0, 1, pressure_scale)
        lift = (lift - lift.mean()) / lift_scale
        pressures.extend([pressure] * (1 + extra_copies))
        lifts.extend([lift] * (1 + extra_copies))
    return np.vstack(pressures), np.concatenate(lifts), wall1, wall2
