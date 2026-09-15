"""Generate cantilever displacements and their three modal-coordinate targets."""

import numpy as np
from scipy.optimize import root_scalar


# Generate all reference modal combinations at the 30 candidate beam positions.
def generate_shm_data():
    """Return scaled displacement, modal coordinates, x/L positions, and mode shapes."""
    positions = np.linspace(1 / 30, 1, 30)
    roots = np.array([root_scalar(lambda beta: np.cosh(beta) * np.cos(beta) + 1,
                                  bracket=(start, start + 3)).root for start in (1, 4, 7)])
    sigma = (np.sinh(roots) - np.sin(roots)) / (np.cosh(roots) + np.cos(roots))
    bx = roots[:, None] * positions
    modes = np.cosh(bx) - np.cos(bx) - sigma[:, None] * (np.sinh(bx) - np.sin(bx))
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)

    # Preserve the reference nesting order: mode 1 outermost and mode 3 innermost.
    coordinates = np.stack(np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 50),
                                      np.linspace(-1, 1, 50), indexing="ij"), axis=-1).reshape(-1, 3)
    displacement = coordinates @ modes
    displacement /= np.max(np.abs(displacement))
    return displacement, coordinates, positions, modes
