import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from sklearn.model_selection import train_test_split
from sklearn.cluster import *
import captum
import pickle
import random
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['text.usetex'] = True
plt.rc('axes', labelsize=20, titlesize=20 )
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)

"""# Utilities"""

def EI_dapr(Xnx, mass_matrix, num_sensors):
    """
    Implement EI-DAPR (Effective Independence with Driving-Point Residue) method

    Parameters:
    Xnx : numpy.ndarray
        Mode shape matrix (num_nodes × num_modes)
    mass_matrix : numpy.ndarray
        Diagonal mass matrix (num_nodes × num_nodes)
    num_sensors : int
        Number of sensors to select

    Returns:
    sensor_locations : list
        Optimal sensor locations (0-based indices)
    """
    # Verify mass matrix is diagonal
    if not np.allclose(mass_matrix, np.diag(np.diag(mass_matrix))):
        raise ValueError("Mass matrix must be diagonal")

    num_nodes = Xnx.shape[0]
    candidates = list(range(num_nodes))
    mass_diag = np.diag(mass_matrix)

    # Precompute Driving-Point Residues (DPR)
    DPR = np.sum((Xnx**2) * mass_diag[:, np.newaxis], axis=1)

    while len(candidates) > num_sensors:
        current_Xnx = Xnx[candidates, :]
        current_mass = mass_diag[candidates]

        # Construct Fisher Information Matrix (FIM) with mass
        FIM = (current_Xnx.T * current_mass) @ current_Xnx

        try:
            FIM_inv = np.linalg.inv(FIM)
        except np.linalg.LinAlgError:
            FIM_inv = np.linalg.pinv(FIM)

        # Calculate Effective Independence (EI)
        Efi = np.array([
            current_mass[i] * current_Xnx[i] @ FIM_inv @ current_Xnx[i].T
            for i in range(len(candidates))
        ])

        # Combine EI with DPR
        combined_scores = Efi * DPR[candidates]

        # Remove node with worst combined score
        candidates.pop(np.argmin(combined_scores))

    return sorted(candidates)

def EI_mass(Xnx, mass_matrix, num_sensors):
    """
    Implement the Effective Independence-Mass (EI-Mass) method for optimal sensor placement

    Parameters:
    Xnx : numpy.ndarray
        Mode shape matrix with shape (num_nodes, num_modes)
    mass_matrix : numpy.ndarray
        Diagonal mass matrix with shape (num_nodes, num_nodes)
    num_sensors : int
        Number of sensors to place

    Returns:
    sensor_locations : list
        Indices of optimal sensor locations (0-based)
    """
    # Verify mass matrix is diagonal
    if not np.allclose(mass_matrix, np.diag(np.diag(mass_matrix))):
        raise ValueError("Mass matrix must be diagonal")

    num_nodes = Xnx.shape[0]
    candidates = list(range(num_nodes))
    mass_diag = np.diag(mass_matrix)

    while len(candidates) > num_sensors:
        # Get current system matrices
        current_Xnx = Xnx[candidates, :]
        current_mass = mass_diag[candidates]

        # Calculate weighted Fisher Information Matrix
        FIM = (current_Xnx.T * current_mass) @ current_Xnx

        try:
            FIM_inv = np.linalg.inv(FIM)
        except np.linalg.LinAlgError:
            FIM_inv = np.linalg.pinv(FIM)

        # Calculate Effective Independence values
        Efi = np.zeros(len(candidates))
        for i, node in enumerate(candidates):
            phi_i = current_Xnx[i]
            Efi[i] = mass_diag[node] * phi_i @ FIM_inv @ phi_i.T

        # Remove node with smallest Efi value
        candidates.pop(np.argmin(Efi))

    return sorted(candidates)

def KE(Xnx, num_sensors):
    """
    Implement the Kinetic Energy method for optimal sensor placement

    Parameters:
    Xnx : numpy.ndarray
        Mode shape matrix with shape (num_nodes, num_modes)
    num_sensors : int
        Number of sensors to place

    Returns:
    sensor_locations : list
        Indices of optimal sensor locations (0-based)
    """
    # Calculate the kinetic energy at each node
    # KE = sum over all modes of (mode shape components squared)
    kinetic_energy = np.sum(Xnx**2, axis=1)

    # Sort nodes by kinetic energy in descending order
    sorted_indices = np.argsort(kinetic_energy)[::-1]

    # Select the top nodes with highest kinetic energy
    sensor_locations = sorted_indices[:num_sensors]

    return sensor_locations

def WAKE(Xnx, num_sensors, mode_weights=None):
    """
    Implement the Weighted Average Kinetic Energy (WAKE) method for optimal sensor placement

    Parameters:
    Xnx : numpy.ndarray
        Mode shape matrix with shape (num_nodes, num_modes)
    num_sensors : int
        Number of sensors to place
    mode_weights : numpy.ndarray or list, optional
        Weights for each mode (default: equal weighting)

    Returns:
    sensor_locations : list
        Indices of optimal sensor locations (0-based)
    """
    # Set default equal weights if none provided
    if mode_weights is None:
        mode_weights = np.ones(Xnx.shape[1])

    # Normalize weights to sum to 1
    mode_weights = np.array(mode_weights) / np.sum(mode_weights)

    # Calculate weighted kinetic energy at each node
    weighted_energy = np.sum((Xnx**2) * mode_weights, axis=1)

    # Sort nodes by weighted kinetic energy in descending order
    sorted_indices = np.argsort(weighted_energy)[::-1]

    # Select the top nodes with highest weighted kinetic energy
    sensor_locations = sorted_indices[:num_sensors]

    return sensor_locations

def process_modes(Xnx, sensor_nodes,
                  L_total=0.45, n_elements=30,
                  rho=5219, A=0.02 * 0.002, verbose=True):
    """
    Processes mode shapes and computes metrics including MMAC, RMS, CN, and DET.

    Parameters:
        Xnx (ndarray): Mode shape matrix (sensors x modes).
        sensor_nodes (list or ndarray): Indices of sensor nodes.
        L_total (float): Total length of the beam [m].
        n_elements (int): Number of finite elements.
        rho (float): Density [kg/m³].
        A (float): Cross-sectional area [m²].

    Returns:
        dict: Dictionary containing mode shapes, MMAC, RMS, CN, and DET.
    """
    L = L_total / n_elements  # Element length

    # Assemble full global mass matrix
    M_global = assemble_global_mass_matrix(n_elements, rho, A, L)

    # Mass normalize mode shapes
    mode_shapes = Xnx.T  # shape: (modes x sensors)
    mode_shapes = mass_normalize_modes(mode_shapes, M_global)
    mode_shapes = mode_shapes[sensor_nodes, :]

    # Apply Guyan reduction
    M_s = compute_guyan_reduction(M_global, sensor_nodes)

    # Compute metrics
    MMAC = compute_mmac(M_s, mode_shapes)
    RMS = compute_RMS(MMAC)
    CN = compute_CN(mode_shapes)
    DET = compute_DET(mode_shapes)

    if verbose:
      print("RMS =", RMS)
      print("CN =", CN)
      print("DET =", DET)

    return {
        "mode_shapes": mode_shapes,
        "MMAC": MMAC,
        "RMS": RMS,
        "CN": CN,
        "DET": DET
    }


def compute_RMS(MMAC):
    """
    Compute the root mean square of the off-diagonals of MMAC matrix.
    """
    n_modes = MMAC.shape[0]
    sum = 0.0

    for i in range(n_modes):
        for j in range(n_modes):
            if i != j:
                sum += MMAC[i, j]**2
    return np.sqrt(sum / (n_modes**2 - n_modes))

def compute_CN(mode_shapes):
    """
    Compute the condition number of the mode shape matrix
    """
    return np.linalg.cond(mode_shapes)

def compute_DET(mode_shapes):
    """
    Compute the determinant of the mode shape matrix
    """
    # for i in range(mode_shapes.shape[0]):
    #     mode_shapes[i, :] /= np.linalg.norm(mode_shapes[i, :])
    # mode_shapes /= np.sqrt(25)
    # mode_shapes /= np.linalg.norm(mode_shapes)
    return np.linalg.det(mode_shapes.T @ mode_shapes)

def compute_mmac(M_s, mode_shapes):
    """
    Compute Mass-Weighted Modal Assurance Criterion (MMAC) matrix.
    Each **row** in mode_shapes is a mode shape at the sensor locations.
    """
    n_modes = mode_shapes.shape[1]
    MMAC = np.zeros((n_modes, n_modes))

    for i in range(n_modes):
        phi_i = mode_shapes[:, i]  # Row i = mode shape i
        for j in range(n_modes):
            phi_j = mode_shapes[:, j]  # Row j = mode shape j
            numerator = (phi_i.T @ M_s @ phi_j)**2
            denominator = (phi_i.T @ M_s @ phi_i) * (phi_j.T @ M_s @ phi_j)
            MMAC[i, j] = numerator / denominator

    return MMAC

def mass_normalize_modes(ModeShapes_sensors, M_s):
    """
    Mass normalizes each mode shape in ModeShapes_sensors using M_s.

    Parameters:
        ModeShapes_sensors (numpy array): Mode shape matrix (size: 2*sensors x n_modes)
        M_s (numpy array): Reduced mass matrix from Guyan reduction (size: 2*sensors x 2*sensors)

    Returns:
        ModeShapes_sensors_normalized (numpy array): Mass-normalized mode shape matrix
    """
    ModeShapes_sensors_normalized = np.copy(ModeShapes_sensors)

    n_modes = ModeShapes_sensors.shape[1]
    for i in range(n_modes):
        phi_i = ModeShapes_sensors[:, i]  # Extract mode shape vector
        modal_mass = phi_i.T @ M_s @ phi_i  # Compute modal mass
        ModeShapes_sensors_normalized[:, i] = phi_i / np.sqrt(modal_mass)  # Normalize

    return ModeShapes_sensors_normalized

def plot_cantilever_sensors(L,n_candidate, sensors):
  x_candidate = np.linspace(L/30, L, n_candidate)
  # Plot the beam
  plt.figure(figsize=(10, 2))
  plt.plot([0, L], [0.2, 0.2], 'k-', linewidth=3)  # Beam (horizontal line)
  plt.plot([0, L], [-0.2, -0.2], 'k-', linewidth=3)  # Beam (horizontal line)

  # Fixed support (like a wall at x = 0)
  plt.plot([0, 0], [-0.8, 0.8], 'k', linewidth=3)  # Thick vertical wall

  # Plot arrows (loads) at x_list
  rank = 1
  for i in sensors:
    # plt.scatter(x_candidate[i], 0, marker='x', s=100, color='black', linewidth=2)
    plt.arrow(x_candidate[i], 0.0, 0, 0.5, head_width=0.15, head_length=0.15, fc='k', ec='k')
    annotation = plt.annotate(str(rank), (x_candidate[i]-0.07, 0.7), fontsize=18, color='black')
    plt.plot([x_candidate, x_candidate], [-0.2, 0.2], 'k', linewidth=2)

    rank += 1

  # Formatting
  plt.xlim(-0.5, L + 0.5)
  plt.ylim(-0.8, 0.8)
  plt.axis('off')
  plt.show()

def effective_independence(Phi, s):
    """
    Implements the Effective Independence (EI) method for optimal sensor placement.

    Parameters:
    Phi : numpy.ndarray
        The mode shape matrix of shape (m, n), where m is the number of candidate DOFs and n is the number of modes.
    s : int
        The desired number of sensors (must be >= number of modes, n)

    Returns:
    selected_indices : list of int
        The indices of the optimal sensor locations (original row indices of Phi)
    """
    m, n = Phi.shape
    if s < n:
        raise ValueError("s must be >= number of modes (columns of Phi)")

    selected_indices = list(range(m))
    current_Phi = Phi.copy()

    while len(selected_indices) > s:
        # Compute Fisher Information Matrix (FIM)
        A = current_Phi.T @ current_Phi
        A_inv = np.linalg.inv(A)
        # Compute Effective Independence matrix
        E = current_Phi @ A_inv @ current_Phi.T
        # Extract diagonal elements (E_d)
        E_d = np.diag(E)
        # Find the index of the DOF with the smallest E_d
        worst_local_idx = np.argmin(E_d)
        # Remove the worst DOF from selected indices and current_Phi
        del selected_indices[worst_local_idx]
        current_Phi = np.delete(current_Phi, worst_local_idx, axis=0)

    return selected_indices

def assemble_global_mass_matrix(n_nodes, rho, A, L):
    """
    Assemble the global mass matrix for a cantilever beam using only translational DOFs.
    """
    # n_nodes = n_elements + 1  # Total number of nodes
    total_dofs = n_nodes  # Only translational DOFs

    M_global = np.zeros((total_dofs, total_dofs))

    m = rho * A * L / 2  # Lumped mass per node

    for e in range(n_nodes-1):
        M_global[e, e] += m
        M_global[e+1, e+1] += m
        M_global[e, e+1] += m / 2
        M_global[e+1, e] += m / 2

    return M_global

def compute_guyan_reduction(M_global, sensor_nodes):
    """
    Apply Guyan reduction to compute the reduced mass matrix M_s (translational DOFs only).
    """
    master_dofs = np.array(sensor_nodes)
    slave_dofs = np.setdiff1d(np.arange(M_global.shape[0]), master_dofs)

    # Partition the mass matrix
    M_mm = M_global[np.ix_(master_dofs, master_dofs)]
    M_ss = M_global[np.ix_(slave_dofs, slave_dofs)]
    M_ms = M_global[np.ix_(master_dofs, slave_dofs)]
    M_sm = M_global[np.ix_(slave_dofs, master_dofs)]

    # Compute the reduced mass matrix using Guyan reduction
    M_s = M_mm - M_ms @ np.linalg.inv(M_ss) @ M_sm
    return M_s

def mass_normalize_modes(ModeShapes_sensors, M_s):
    """
    Mass normalizes each mode shape using M_s (translational DOFs only).
    """
    ModeShapes_sensors_normalized = np.copy(ModeShapes_sensors)

    for i in range(ModeShapes_sensors.shape[1]):
        phi_i = ModeShapes_sensors[:, i]
        modal_mass = phi_i.T @ M_s @ phi_i
        ModeShapes_sensors_normalized[:, i] = phi_i / np.sqrt(modal_mass)

    return ModeShapes_sensors_normalized

"""# Data Generation"""

# Parameters
HMMS = 3    # number of modes
L = 1   # cantilever length
n_candidate = 30    # number of nodes

Nm = 3 * HMMS
# Solve for betaNL
betaNL = np.zeros(Nm)
jj = 0
while jj < Nm:
    result = root_scalar(lambda betaNL: np.cosh(betaNL) * np.cos(betaNL) + 1, bracket=[jj + 1, jj + 4])
    betaNL[jj] = result.root
    jj += 3

# Filter non-zero solutions
betaNLall = betaNL[betaNL != 0]
betaN = betaNLall / L

# Spatial domain
# x = np.linspace(0, L, 180)

x = np.linspace(L/30, L, n_candidate)
xl = x / L

# Calculate sigmaN
sigmaN = np.zeros(HMMS)
for ii in range(HMMS):
    sigmaN[ii] = (np.sinh(betaN[ii] * L) - np.sin(betaN[ii] * L)) / (np.cosh(betaN[ii] * L) + np.cos(betaN[ii] * L))

# Compute mode shapes Xnx
Xnx = np.zeros((len(betaN), len(x)))

for ii in range(len(betaN)):
    for jj in range(len(x)):
        cosh_cos = np.cosh(betaN[ii] * x[jj]) - np.cos(betaN[ii] * x[jj])
        sinh_sin = np.sinh(betaN[ii] * x[jj]) - np.sin(betaN[ii] * x[jj])
        Xnx[ii, jj] = cosh_cos - sigmaN[ii] * sinh_sin

Xnx = Xnx/np.expand_dims((np.linalg.norm(Xnx,axis=1)),axis=1)   # Mode shapes normalized by 2-norm

# Data generation

# How many evenly spaced modal coefficients to generate
n_sample = 20   
n_sample2 = 50
n_sample3 = 50
mode1_list = np.linspace(-1, 1, n_sample)
mode2_list = np.linspace(-1, 1, n_sample2)
mode3_list = np.linspace(-1, 1, n_sample3)

N_sample = len(mode1_list)*len(mode2_list)*len(mode3_list)

# save all coordinate combinations and the corresponding displacement profiles
mode_coord = np.zeros((N_sample,3))
disp_all = np.zeros((N_sample,n_candidate))
index = 0
for mode1 in mode1_list:
    for mode2 in mode2_list:
        for mode3 in mode3_list:
          mode_coord_temp = np.array([mode1, mode2, mode3])
          mode_coord[index,:] = mode_coord_temp

          disp_all[index,:] = mode_coord_temp @ Xnx

          index += 1

print("mode_coord:",mode_coord.shape)
print("disp_all:",disp_all.shape)

# Global scaling
disp_max = np.amax(abs(disp_all))
disp_scaled = disp_all/disp_max

# Plot the displacement profiles of the modes
mode_coord_test = [1,1,1]
plt.figure(figsize=(10, 7))
plt.grid(True)
plt.plot(xl, Xnx[0, :], 'b-', linewidth=1.5, label='Mode #1')
plt.plot(xl, Xnx[1, :], 'r-', linewidth=1.5, label='Mode #2')
plt.plot(xl, Xnx[2, :], 'm-', linewidth=1.5, label='Mode #3')

# plot superposed profile of the given coordinates
superposed = mode_coord_test @ Xnx
plt.plot(xl, superposed, '--', color='black', linewidth=2, label='Superposed')

# plot formatting
plt.title('Mode Shapes of the Cantilever Beam')
plt.legend(loc='lower left',fontsize=20)
plt.xlabel('x/L')
plt.ylabel('Displacement $u/\max(u)$')
plt.ylim([-1,1])
plt.xlim([0,1])
plt.show()

"""# Clustering"""

# AffinityPropagation Clustering
aff_matrix = np.abs(np.corrcoef(disp_all.T)) # Compute affinity matrix using correlation coefficient (n_candidates X n_points)
af = AffinityPropagation(damping=0.5,max_iter=10000, convergence_iter=10, copy=True, preference= 0.991, affinity='precomputed').fit(aff_matrix)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(set(labels))
# sensors = cluster_centers_indices
print('n_clusters =',n_clusters_)

# plot cluster results on a cantilever beam
L = 6
x_candidate = np.linspace(L/30, L, n_candidate)
# Plot the beam
plt.figure(figsize=(10, 2))
plt.plot([0, L], [0.2, 0.2], 'k-', linewidth=3)  # Beam (horizontal line)
plt.plot([0, L], [-0.2, -0.2], 'k-', linewidth=3)  # Beam (horizontal line)

# Fixed support (like a wall at x = 0)
plt.plot([0, 0], [-0.8, 0.8], 'k', linewidth=3)  # Thick vertical wall

# Plot arrows (loads) at x_list
color_list = sns.color_palette(palette=cc.glasbey_category10,n_colors=n_clusters_)

for i in range(n_candidate):
  # plt.scatter(x_candidate[i], 0, marker='x', s=100, color='black', linewidth=2)
  # print(color_list[labels[i]])
  if i in cluster_centers_indices:
    plt.arrow(x_candidate[i], 0.0, 0, 0.5, head_width=0.15, head_length=0.15, fc=color_list[labels[i]], ec='k',linewidth=2.0)
  else:
    plt.arrow(x_candidate[i], 0.0, 0, 0.5, head_width=0.15, head_length=0.15, fc=color_list[labels[i]], ec=color_list[labels[i]])
  # annotation = plt.annotate(str(rank), (x_candidate[i]-0.07, 0.7), fontsize=18, color='black')
  plt.plot([x_candidate, x_candidate], [-0.2, 0.2], 'k', linewidth=2)

# Formatting
plt.xlim(-0.5, L + 0.5)
plt.ylim(-0.8, 0.8)
plt.axis('off')
# plt.savefig("SHM_clusters.png",format='png',bbox_inches='tight')
# plt.savefig("SHM_clusters.eps",format='eps')
plt.show()

"""# Model Training and Attribution"""


n_iter = 1  # how many training runs to average
n_sensors = 15 # number of desired sensors
test_split = 0.1  # how much data used for testing

# SET Testing Sensors HERE
# sensors = np.arange(n_candidate)
sensors = cluster_centers_indices

mode = 'test' # 'test' mode evaluates the performance of the given "sensors"
# mode = 'attr' # 'attr' mode performs attribution to rank the candidate sensors
#-------------------------------------------------------------------------------
best_error = 0.01
errors = np.zeros([n_iter,1])
weights_mean_all = np.zeros([n_iter,len(sensors)])
for iter in range(n_iter):
  print('Starting run', iter)
  random_state = np.random.randint(0, 201)
  X_train, X_test, y_train, y_test = train_test_split(
      disp_scaled[:,sensors],
      mode_coord,
      test_size=test_split,
      shuffle=True,
      random_state=random_state
  )

  # make datasets and dataloaders
  train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
  test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

  # set random seed
  torch.manual_seed(random_state)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

  layer_size = 12
  model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(len(sensors), layer_size,bias=True),
        nn.BatchNorm1d(layer_size),
        nn.ReLU(),
        nn.Linear(layer_size, layer_size,bias=True),
        nn.BatchNorm1d(layer_size),
        nn.ReLU(),
        nn.Linear(layer_size, layer_size,bias=True),
        nn.BatchNorm1d(layer_size),
        nn.ReLU(),
        nn.Linear(layer_size, 3,bias=False),
  )

  model = model.to(torch.double)
  print(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.000015, weight_decay=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-7, eps=1e-7, patience=5)
  loss_fn = nn.MSELoss()
  threshold = 1e-5   # early stopping threshold for test loss
  n_epochs_min = 30
  n_epochs_max = 150
  min_loss = 1.0
  print_interval = 1

  # store metrics
  training_loss_history = np.zeros([n_epochs_max, 1])
  validation_loss_history = np.zeros([n_epochs_max, 1])

  for epoch in range(n_epochs_max):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
          optimizer.zero_grad()

          # Forward pass
          output = model(data)

          # Calculate loss
          loss = loss_fn(output, target)

          # Backward pass
          loss.backward()

          # Weight update
          optimizer.step()

          training_loss_history[epoch] += loss_fn(output, target).item()

      # Track loss each epoch
      training_loss_history[epoch] /= len(train_loader)
      if epoch % print_interval == 0:
        print(f'Train Epoch: %d/%i  Train Loss: %1.4e' % (epoch + 1, n_epochs_max, training_loss_history[epoch,0]),end='')


      # Turning off automatic differentiation
      with torch.no_grad():
          model.eval()
          for data, target in test_loader:
              # target = target.unsqueeze(1)
              output = model(data)
              validation_loss_history[epoch] += loss_fn(output, target).item()  # Sum up batch loss

          validation_loss_history[epoch] /= len(test_loader)
          if epoch % print_interval == 0:
            print(f', Test Loss: %1.4e' % validation_loss_history[epoch,0],end='')

      # early stop if a new minimum loss is reached after "n_epochs_min" epochs
      if epoch > n_epochs_min:
        if validation_loss_history[epoch] < min_loss:
          break
      else:
          if validation_loss_history[epoch] < min_loss:
            min_loss = validation_loss_history[epoch]

      scheduler.step(validation_loss_history[epoch,0]) # adjust learning rate
      if epoch % print_interval == 0:
        print(f', Learning Rate = %1.2e' % optimizer.param_groups[0]['lr'])

  # Plot loss history
  plt.rcParams['figure.figsize'] = [6, 5] # change figure size
  plt.plot(validation_loss_history[2:epoch], color='b',linewidth=2, label='Test Loss')
  plt.plot(training_loss_history[2:epoch],':',color='r',linewidth=2, label='Train Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

# save the model that gives the lowest validation loss
  if validation_loss_history[epoch] < best_error:
    best_error = validation_loss_history[epoch]
    best_model = model

  if mode == 'test':
    with torch.no_grad():
      test_output = model(torch.tensor(X_test)).detach().numpy()
    errors[iter,0] = np.linalg.norm(test_output - y_test,2)/np.linalg.norm(y_test - np.mean(y_test),2) # L2 error norm
  else:
    # perform attribution using the last trained model
    # Find most important sensors
    ig = captum.attr.IntegratedGradients(model)
    # randomly select 20000 samples from 1 to n_sample^3
    samplei = random.sample(range(N_sample), 20000)
    disp_temp = disp_scaled[samplei,:]
    test_shap = torch.tensor(disp_temp[:,sensors])
    test_shap.requires_grad_()
    attr = ig.attribute(test_shap,target=0,n_steps=50)
    attr = attr.detach().numpy()
    weights_mean_all[iter,:] = np.mean(abs(attr),axis=0) # average over time samples (1st axis)

if mode == 'test':
  print('L2 errors =', errors)
  print('Average L2 error =', np.mean(errors))
else:
  weights_mean = np.mean(weights_mean_all,axis=0) # average over iterations
  sensor_temp = np.flip(np.argsort(weights_mean))
  attr_sensors = sensors[sensor_temp[:n_sensors]] # Indices of the attribution sensor locations
  sensors_str = ','.join(map(str, attr_sensors)) #print "sensors" array as string and add commas
  print(str(n_sensors)+" attribution sensors:",sensors_str)
  results = process_modes(Xnx, attr_sensors[:5], verbose = True)

#attr_sensors = [25,9,29,27,11,24,21,13,17,23,12,19,14,16,7,22,15,26,3] # 19 attr sensors, 19 candidates, 30k IG points

# plot optimal sensors and compute the performance metrics
plot_cantilever_sensors(6,30, attr_sensors[:5])
results = process_modes(Xnx, attr_sensors[:5], verbose = True)

plot_cantilever_sensors(6,30, attr_sensors[:15])
results = process_modes(Xnx, attr_sensors[:15], verbose = True)

# compute and plot performance as a functino of sensor number
n_sensor_max = 19
n_sensor_min = 5
n_sensors_list = np.arange(n_sensor_min,n_sensor_max+1)
RMS = np.zeros(n_sensor_max-n_sensor_min+1)
CN = np.zeros(n_sensor_max-n_sensor_min+1)
DET = np.zeros(n_sensor_max-n_sensor_min+1)

RMS_EI = np.zeros(n_sensor_max-n_sensor_min+1)
CN_EI = np.zeros(n_sensor_max-n_sensor_min+1)
DET_EI = np.zeros(n_sensor_max-n_sensor_min+1)

RMS_KE = np.zeros(n_sensor_max-n_sensor_min+1)
CN_KE = np.zeros(n_sensor_max-n_sensor_min+1)
DET_KE = np.zeros(n_sensor_max-n_sensor_min+1)

for n_sensors in n_sensors_list:
  sensors_test = attr_sensors[:n_sensors]
  results = process_modes(Xnx, sensors_test, verbose = False)
  RMS[n_sensors-n_sensor_min] = results['RMS']
  CN[n_sensors-n_sensor_min] = results['CN']
  DET[n_sensors-n_sensor_min] = results["DET"]

  EI_sensors = effective_independence(Xnx.T, n_sensors)
  results = process_modes(Xnx, EI_sensors, verbose = False)
  RMS_EI[n_sensors-n_sensor_min] = results['RMS']
  CN_EI[n_sensors-n_sensor_min] = results['CN']
  DET_EI[n_sensors-n_sensor_min] = results["DET"]

  KE_sensors = KE(Xnx.T, n_sensors)
  results = process_modes(Xnx, KE_sensors, verbose = False)
  RMS_KE[n_sensors-n_sensor_min] = results['RMS']
  CN_KE[n_sensors-n_sensor_min] = results['CN']
  DET_KE[n_sensors-n_sensor_min] = results["DET"]

plt.figure(figsize=(10, 7))
plt.plot(n_sensors_list,RMS, color='b',linewidth=2, label='RMS')
plt.plot(n_sensors_list,RMS_EI, color='r',linewidth=2, label='RMS_EI')
plt.plot(n_sensors_list,RMS_KE, color='g',linewidth=2, label='RMS_KE')
plt.xlabel('$n_{sensor}$',fontsize=25)
plt.ylabel('RMS')
plt.xlim([n_sensor_min,n_sensor_max])
plt.xticks(range(n_sensor_min, n_sensor_max+1, 2))
# plt.savefig('SHM_RMS.eps',format = 'eps')
# plt.savefig('SHM_RMS.png')
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(n_sensors_list,CN,color='b',linewidth=2, label='CN')
plt.plot(n_sensors_list,CN_EI,color='r',linewidth=2, label='CN_EI')
plt.plot(n_sensors_list,CN_KE,color='g',linewidth=2, label='CN_KE')
plt.xlabel('$n_{sensor}$',fontsize=25)
plt.ylabel('CN')
plt.ylim([1,5])
plt.xlim([n_sensor_min,n_sensor_max])
plt.xticks(range(n_sensor_min, n_sensor_max+1, 2))
# plt.savefig('SHM_CN.eps',format = 'eps')
# plt.savefig('SHM_CN.png')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(n_sensors_list,DET,color='b',linewidth=2, label='DET')
plt.plot(n_sensors_list,DET_EI,color='r',linewidth=2, label='DET_EI')
plt.plot(n_sensors_list,DET_KE,color='g',linewidth=2, label='DET_KE')
plt.xlabel('$n_{sensor}$',fontsize=25)
plt.ylabel('DET')
plt.xlim([n_sensor_min,n_sensor_max])
plt.xticks(range(n_sensor_min, n_sensor_max+1, 2))
# plt.savefig('SHM_DET.eps',format = 'eps')
# plt.savefig('SHM_DET.png')
plt.show()