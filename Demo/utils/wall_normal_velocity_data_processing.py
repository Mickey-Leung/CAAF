"""Load pressure patches and separate velocity-probe histories from MATLAB data."""

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


# Read only the required velocity plane and retain each probe as a separate trajectory.
def load_wall_normal_velocity_data(data_dir, *, x_indices=range(10, 200, 10),
                                  z_indices=range(10, 40, 10), wall_normal_index=22,
                                  half_window=9, re_tau=186.0):
    """Return pressure X, velocity y, (x+, z+) sensor offsets, and trajectory IDs."""
    data_dir = Path(data_dir)
    required = ("V_data.mat", "P_w.mat", "x_edge.mat", "y_edge.mat", "z_edge.mat")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing velocity demo files in {data_dir}: {', '.join(missing)}.")
    for value, name in ((half_window, "half_window"), (wall_normal_index, "wall_normal_index")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    x_indices, z_indices = np.asarray(list(x_indices)), np.asarray(list(z_indices))
    for indices in (x_indices, z_indices):
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu" or np.any(indices < half_window):
            raise ValueError("Probe indices must be nonempty integer lists with room for each pressure patch.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Probe indices must be unique within each axis.")
    if not np.isfinite(re_tau) or re_tau <= 0:
        raise ValueError("re_tau must be finite and positive.")
    x_stop, z_stop = int(x_indices.max()) + half_window + 1, int(z_indices.max()) + half_window + 1
    x = loadmat(data_dir / "x_edge.mat")["x"].reshape(-1)
    y_grid = loadmat(data_dir / "y_edge.mat")["y"].reshape(-1)
    z = loadmat(data_dir / "z_edge.mat")["z"].reshape(-1)
    if len(x) < max(2, x_stop) or len(z) < max(2, z_stop) or len(y_grid) <= wall_normal_index:
        raise ValueError("Coordinate arrays do not cover the requested pressure patches and velocity plane.")
    dx, dz = x[1] - x[0], z[1] - z[0]
    if not np.isfinite([y_grid[wall_normal_index], dx, dz]).all() or dx <= 0 or dz <= 0 or not (
            np.allclose(np.diff(x[:x_stop]), dx) and np.allclose(np.diff(z[:z_stop]), dz)):
        raise ValueError("The selected pressure grid must have finite, increasing, uniform x and z spacing.")

    # MATLAB v7.3 stores reversed axes: velocity (z, y, x, time), pressure (z, x, time).
    with h5py.File(data_dir / "V_data.mat", "r") as velocity_file, h5py.File(data_dir / "P_w.mat", "r") as pressure_file:
        if "V_data" not in velocity_file or "P_w" not in pressure_file:
            raise ValueError("MATLAB files must contain the V_data and P_w datasets, respectively.")
        velocity_data, pressure_data = velocity_file["V_data"], pressure_file["P_w"]
        if velocity_data.ndim != 4 or pressure_data.ndim != 3:
            raise ValueError("Expected reversed MATLAB axes: V_data (z, y, x, time), P_w (z, x, time).")
        if (velocity_data.shape[-1] != pressure_data.shape[-1] or pressure_data.shape[-1] < 1
                or velocity_data.shape[1] <= wall_normal_index
                or min(velocity_data.shape[0], pressure_data.shape[0]) < z_stop
                or min(velocity_data.shape[2], pressure_data.shape[1]) < x_stop):
            raise ValueError("Pressure and velocity dimensions must cover the probes and have matching time counts.")
        velocity = velocity_data[:z_stop, wall_normal_index, :x_stop, :].transpose(2, 1, 0)
        pressure = pressure_data[:z_stop, :x_stop, :].transpose(2, 1, 0)

    # Flatten each patch with streamwise position varying fastest, matching the reference.
    nt = len(velocity)
    width = 2 * half_window + 1
    probes = [(int(ix), int(iz)) for ix in x_indices for iz in z_indices]
    X = np.empty((nt * len(probes), width ** 2), dtype=np.float64)
    targets = np.empty(nt * len(probes), dtype=np.float64)
    for group, (ix, iz) in enumerate(probes):
        rows = slice(group * nt, (group + 1) * nt)
        X[rows] = pressure[:, ix-half_window:ix+half_window+1,
                          iz-half_window:iz+half_window+1].reshape(nt, width ** 2, order="F")
        targets[rows] = velocity[:, ix, iz]
    if not np.isfinite(X).all() or not np.isfinite(targets).all():
        raise ValueError("Pressure patches and velocity targets must contain only finite values.")
    offsets = np.arange(-half_window, half_window + 1)
    x_grid, z_grid = np.meshgrid(offsets * dx * re_tau, offsets * dz * re_tau)
    sensor_coordinates = np.column_stack((x_grid.ravel(), z_grid.ravel()))
    groups = np.repeat(np.arange(len(probes)), nt)
    return X, targets, sensor_coordinates, groups
