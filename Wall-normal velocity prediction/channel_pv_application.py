import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import seaborn as sns
import colorcet as cc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import *
import captum
import scipy.io
import h5py

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["mathtext.fontset"]= 'cm'  # Use Computer Modern for math
plt.rc('axes', labelsize=28, titlesize=28)
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)

"""# Data Setup"""

# Choose normalization method: "01" for [0,1]; "m11" for [-1,1]; "u_tau" for dimensionless
normalization = "m11"

# Load Datasets
nu = 3.076923076923077e-04
utau = 0.057231
Retau = 186
dt_plus = utau**2/nu*0.1
with h5py.File('V_data.mat', 'r') as file:
  V_raw = file['V_data'][()].T # Access the 'V_probe' dataset and convert to NumPy array
with h5py.File('P_w.mat', 'r') as file:
  P_w_raw = file['P_w'][()].T # Access the 'V_probe' dataset and convert to NumPy array
# (nt, nx, ny, nz)

ix_list = range(200) # X indices of wall shear stress probes
iy_list = 22 # y+=10
iz_list = range(40)


data = scipy.io.loadmat('x_edge.mat')
x_edge = data['x']
data = scipy.io.loadmat('y_edge.mat')
y_edge = data['y']
data = scipy.io.loadmat('z_edge.mat')
z_edge = data['z']

x_plus = x_edge[ix_list]*Retau
y_plus = y_edge*Retau
z_plus = z_edge[iz_list]*Retau

nt,nx,ny,nz = V_raw.shape

V_xz = V_raw[:,:,iy_list,:] # xz slice of V at y+=9.6
ix_samples = np.arange(10,200,10) # X indices of velocity probes
iz_samples = np.arange(10,40,10)  # Z indices of velocity probes
x_window = 9 # how many grid points to consider in one direction
z_window = 9

# sample and stack probe and target data
P_w_probe_raw  = np.zeros((0,x_window*2+1,z_window*2+1))
V_probe_raw = np.zeros((0))

for ix in ix_samples:
  for iz in iz_samples:
    V_probe_raw = np.concatenate((V_probe_raw,V_xz[:,ix,iz]))
    P_w_probe_raw = np.concatenate((P_w_probe_raw,P_w_raw[:,ix-x_window:ix+x_window+1,iz-z_window:iz+z_window+1]),axis=0)
print("V_probe_raw", V_probe_raw.shape)
print("P_w_probe_raw", P_w_probe_raw.shape)

n_points, n_xprobe, n_zprobe = P_w_probe_raw.shape
print('n_points =',n_points)
n_probe = n_xprobe*n_zprobe
print('n_probe =',n_probe)

dx = np.ediff1d(x_plus)[0]
dz = dx

X_probe = np.arange(-x_window, x_window+1,)*dx
Z_probe = np.arange(-z_window, z_window+1)*dz

x_grid, z_grid = np.meshgrid(X_probe, Z_probe) # (nz,nx) reverse dimensions
x_flatten = x_grid.reshape(n_probe)
z_flatten = z_grid.reshape(n_probe)
P_flatten = P_w_probe_raw.reshape(n_points,n_probe,order='F')
print('n_probe =',n_probe)

def normalize_01(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_scaled = (data - data_min) / (data_max - data_min)
    return data_scaled, data_min, data_max

def unnormalize_01(data_scaled, data_min, data_max):
    return data_scaled * (data_max - data_min) + data_min

def normalize_m1_1(data):
    data_mean = np.mean(data, axis=0)
    max_range = np.max(np.abs(data - data_mean), axis=0)
    data_scaled = (data - data_mean) / max_range
    return data_scaled, data_mean, max_range

def unnormalize_m1_1(data_scaled, data_mean, max_range):
    return data_scaled * max_range + data_mean

def normalize_utau(data, utau, scale=0.1, square=False):
    data_mean = np.mean(data, axis=0)
    denom = utau**2 if square else utau
    data_scaled = scale * (data - data_mean) / denom
    return data_scaled, data_mean

def unnormalize_utau(data_scaled, data_mean, utau, scale=0.1, square=False):
    denom = utau**2 if square else utau
    return data_scaled * denom / scale + data_mean

def normalize_utau_01(data, utau, scale=1.0, square=False):
    # Step 1: Normalize by u_tau
    data_mean = np.mean(data, axis=0)
    denom = utau**2 if square else utau
    data_utau = scale * (data - data_mean) / denom

    # Step 2: Normalize to [0, 1]
    data_min = np.min(data_utau, axis=0)
    data_max = np.max(data_utau, axis=0)
    data_scaled = (data_utau - data_min) / (data_max - data_min)

    return data_scaled, data_mean, data_min, data_max

def unnormalize_utau_01(data_scaled, data_mean, data_min, data_max, utau, scale=1.0, square=False):
    # Step 1: Unscale from [0, 1]
    data_utau = data_scaled * (data_max - data_min) + data_min

    # Step 2: Unnormalize from u_tau
    denom = utau**2 if square else utau
    data_original = data_utau * denom / scale + data_mean

    return data_original

# # For [0, 1] normalization
if normalization == "01":
  P_scaled, P_min, P_max = normalize_01(P_flatten)
  V_scaled, V_min, V_max = normalize_01(V_probe_raw)
elif normalization == "m11":
  # For [-1, 1] normalization
  P_scaled, P_mean, P_range = normalize_m1_1(P_flatten)
  V_scaled, V_mean, V_range = normalize_m1_1(V_probe_raw)
elif normalization == "u_tau":
  # For u_tau normalization
  P_scaled, P_mean = normalize_utau(P_flatten, utau, scale=0.1, square=True)
  V_scaled, V_mean = normalize_utau(V_probe_raw, utau, scale=1.0, square=False)
elif normalization == "u_tau_01":
  # Normalize V using u_tau and then to [0,1]
  P_scaled, P_mean, P_min, P_max = normalize_utau_01(P_flatten, utau, scale=1.0, square=True)
  V_scaled, V_mean, V_min, V_max = normalize_utau_01(V_probe_raw, utau, scale=1.0, square=False)
elif normalization == "standard":
  # For standard normalization
  scaler = StandardScaler()
  P_scaled = scaler.fit_transform(P_flatten)
  V_scaled = scaler.fit_transform(V_probe_raw.reshape(-1, 1)).reshape(-1)

print(P_scaled.shape)
print(V_scaled.shape)

P_eval = P_scaled[26100:27000,:]
V_eval = V_scaled[26100:27000]

# remove the evaluation points from P_scaled and V_scaled
P_scaled_train = np.delete(P_scaled, np.s_[26100:27000], axis=0)
V_scaled_train = np.delete(V_scaled, np.s_[26100:27000])

print(P_scaled_train.shape)
print(V_scaled_train.shape)
print(P_eval.shape)
print(V_eval.shape)

fig = plt.figure(figsize=(8,8))
plt.rc('axes', labelsize=28, titlesize=28)
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)

# Create scatter plots for all points
plt.scatter(x_flatten, z_flatten, s=20, marker='o', c='black')
plt.axvline(x=0, color='black', linestyle=':')
plt.axhline(y=0, color='black', linestyle=':')

plt.ylim([-50,50])
plt.yticks([-50,-25,0,25,50])
plt.xlim([-50,50])
plt.xticks([-50,-25,0,25,50])
plt.xlabel('$\Delta x^+$')
plt.ylabel('$\Delta z^+$')
plt.title('Probe Locations')
#plt.savefig("clusters_map.png",format='png',bbox_inches='tight')
plt.show()

"""# Remove Correlated Sensors"""

# Plot 2d contour for using V_probe_raw[1,:,:].T, x_grid, z_grid

fig, ax = plt.subplots(figsize=(8, 8))
contour_levels = np.linspace(P_w_probe_raw.min(), P_w_probe_raw.max(), 50)
cf = ax.contourf(x_grid, z_grid, P_w_probe_raw[500, :, :].T, levels=contour_levels, cmap='viridis')
fig.colorbar(cf, ax=ax, label='$V$')
ax.set_xlabel('$\Delta x^+$')
ax.set_ylabel('$\Delta z^+$')
ax.set_title('Contour of $P$')
plt.show()

# # AffinityPropagation Clustering
aff_matrix = np.abs(np.corrcoef(P_flatten.T)) # Compute affinity matrix using correlation coefficient
af = AffinityPropagation(damping=0.5,max_iter=10000, convergence_iter=50, copy=True, preference= 0.9, affinity='precomputed').fit(aff_matrix)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(set(labels))
print('n_clusters =',n_clusters_)

fig = plt.figure(figsize=(8, 8))
color_list = sns.color_palette(palette=cc.glasbey_category10,n_colors=n_clusters_)

# Create scatter plots for all points
plt.axvline(x=0, color='black', linestyle=':')
plt.axhline(y=0, color='black', linestyle=':')
plt.scatter(x_flatten, z_flatten, s=20, marker='o', c=[color_list[i] for i in labels])
plt.scatter(x_flatten[cluster_centers_indices],z_flatten[cluster_centers_indices],marker='o', s=90, linewidths=2, color='black',facecolors='none')

plt.ylim([-50,50])
plt.yticks([-50,-25,0,25,50])
plt.xlim([-50,50])
plt.xticks([-50,-25,0,25,50])
plt.xlabel(r'$\Delta x^+$', fontsize=25)
plt.ylabel(r'$\Delta z^+$', fontsize=25)
plt.tick_params(axis='both', labelsize=22)
plt.savefig("PV_clusters.png",format='png',bbox_inches='tight')
plt.savefig("PV_clusters.pdf",format='pdf',bbox_inches='tight')
plt.show()

"""# Average Model Training and Attribution"""

# SET Testing Sensors HERE

n_iter = 1
n_sensors = 10
test_split = 0.1
# sensors = np.array([182,187,129,139,106,243,231,204,93,260,283,267,226,103,134,300,218,294,63,197]) # 20 CAAF best so far
# sensors = sensors[:n_sensors]

# sensors = np.random.choice(cluster_centers_indices, size=n_sensors, replace=False)
sensors = cluster_centers_indices
# sensors = np.arange(n_probe)


mode = 'test'
# mode = 'attr'

use_GPU = True
#-------------------------------------------------------------------------------
best_model = None
best_error = 10.0
errors = np.zeros([n_iter,1])
weights_mean_all = np.zeros([n_iter,len(sensors)])
for iter in range(n_iter):
  print('Starting run', iter)
  # random_state = np.random.randint(0, 201)
  random_state = 139
  print('Random State =',random_state)
  X_train, X_test, y_train, y_test = train_test_split(
      P_scaled_train[:,sensors],
      V_scaled_train,
      test_size=test_split,
      shuffle=False,
      random_state=random_state
  )

  # make datasets and dataloaders
  train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
  test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

  # set random seed
  torch.manual_seed(random_state)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

  if use_GPU:
    torch.cuda.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

  # Best model for 58 cluster centers
  if len(sensors) > 30:
    # model for CAAF
    layer_size = 128
    model = nn.Sequential(
          nn.Flatten(),
          nn.Linear(len(sensors), layer_size,bias=True),
          nn.BatchNorm1d(layer_size),
          nn.LeakyReLU(),
          nn.Linear(layer_size, layer_size,bias=True),
          nn.BatchNorm1d(layer_size),
          nn.LeakyReLU(),
          nn.Linear(layer_size, 1,bias=True),
    )
  else:
    # model for evaluation
    layer_size = 50
    model = nn.Sequential(
          nn.Flatten(),
          nn.Linear(len(sensors), layer_size,bias=True),
          nn.BatchNorm1d(layer_size),
          nn.LeakyReLU(),
          nn.Linear(layer_size, layer_size,bias=True),
          nn.BatchNorm1d(layer_size),
          nn.LeakyReLU(),
          nn.Linear(layer_size, layer_size,bias=True),
          nn.BatchNorm1d(layer_size),
          nn.LeakyReLU(),
          nn.Linear(layer_size, 1,bias=True),
    )

  model = model.to(torch.double)
  print(model)

  if use_GPU:
    model = model.to(device)

  print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

  optimizer = torch.optim.Adam(model.parameters(), lr=0.00003, weight_decay=1e-2)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-7, eps=1e-6, patience=2) # smaller eps more sensitive
  loss_fn = nn.MSELoss()
  threshold = 1e-3   # early stopping threshold for test loss
  n_epochs_min = 40
  n_epochs_max = 200
  min_loss = 10.0
  print_interval = 1
  early_stop_patience = 20
  early_stop_counter = 0

  # store metrics
  training_loss_history = np.zeros([n_epochs_max, 1])
  validation_loss_history = np.zeros([n_epochs_max, 1])
  corr_history = np.zeros(n_epochs_max)

  for epoch in range(n_epochs_max):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
          if use_GPU:
            data, target = data.to(device), target.to(device)
          target = target.unsqueeze(1)
          # Erase accumulated gradients
          optimizer.zero_grad()

          # Forward pass
          output = model(data)

          # Calculate loss
          loss = loss_fn(output, target)

          # Backward pass
          loss.backward()

          # Weight update
          optimizer.step()

          training_loss_history[epoch] += nn.MSELoss()(output, target).item()

      # Track loss each epoch
      training_loss_history[epoch] /= len(train_loader)
      if epoch % print_interval == 0:
        print(f'Train Epoch: %d/%i  Train Loss: %1.4e' % (epoch + 1, n_epochs_max, training_loss_history[epoch,0]),end='')

      # Turning off automatic differentiation
      with torch.no_grad():
          model.eval()
          for data, target in test_loader:
              if use_GPU:
                data, target = data.to(device), target.to(device)
              target = target.unsqueeze(1)
              output = model(data)
              validation_loss_history[epoch] += nn.MSELoss()(output, target).item()  # Sum up batch loss

              # compute correlation
              corr_epoch = np.corrcoef(output.cpu().detach().numpy().reshape((-1)), target.cpu().detach().numpy().reshape((-1)))[0,1]
              corr_history[epoch] += corr_epoch

          validation_loss_history[epoch] /= len(test_loader)
          corr_history[epoch] /= len(test_loader)
          if epoch % print_interval == 0:
            print(f', Test Loss: %1.4e' % validation_loss_history[epoch,0],end='')
            print(f', Corr: %1.4e' % corr_history[epoch],end='')

      loss_rate = (validation_loss_history[epoch-1]-validation_loss_history[epoch])/validation_loss_history[epoch]
      # loss_rate = (training_loss_history[epoch-1]-training_loss_history[epoch])/training_loss_history[epoch]
      if epoch % print_interval == 0:
        print(f', Loss Rate = %1.2e' % loss_rate[0],end='')
      # if epoch > n_epochs_min and loss_rate < threshold and loss_rate > 0: #loss_rate > 0 and
      #   break
      # if loss_rate < 0: early_stop_counter += 1
      # if epoch > n_epochs_min and ((early_stop_counter > early_stop_patience) or (loss_rate < threshold and loss_rate > 0)): #stop if loss_rate > 0 and < threshold
      #   break

      scheduler.step(validation_loss_history[epoch,0]) # adjust learning rate
      if epoch % print_interval == 0:
        print(f', Learning Rate = %1.2e' % optimizer.param_groups[0]['lr'])

  plt.rcParams['figure.figsize'] = [6, 5] # change figure size
  plt.plot(validation_loss_history[2:epoch], color='b',linewidth=2, label='Test Loss')
  plt.plot(training_loss_history[2:epoch],':',color='r',linewidth=2, label='Train Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()


  if mode == 'test':
    with torch.no_grad():
      if use_GPU:
        X_test_temp = torch.tensor(X_test).to(device)
      else:
        X_test_temp = torch.tensor(X_test)
      test_output = model(X_test_temp).cpu().detach().numpy()

    if normalization == "01":
      test_output = unnormalize_01(test_output, V_min, V_max)
      y_test_unscaled = unnormalize_01(y_test, V_min, V_max)
    elif normalization == "m11":
      test_output = unnormalize_m1_1(test_output, V_mean, V_range)
      y_test_unscaled = unnormalize_m1_1(y_test, V_mean, V_range)
    elif normalization == "u_tau":
      test_output = unnormalize_utau(test_output, V_mean, utau, scale=1.0, square=False)
      y_test_unscaled = unnormalize_utau(y_test, V_mean, utau, scale=1.0, square=False)
    elif normalization == "u_tau_01":
      test_output = unnormalize_utau_01(test_output, V_mean, V_min, V_max, utau, scale=1.0, square=False)
      y_test_unscaled = unnormalize_utau_01(y_test, V_mean, V_min, V_max, utau, scale=1.0, square=False)
    elif normalization == "standard":
      test_output = scaler.inverse_transform(test_output.reshape(-1, 1)).reshape(-1)
      y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # compute L2 error
    errors[iter,0] = np.linalg.norm(test_output.reshape((-1)) - y_test_unscaled,2)/np.linalg.norm(y_test_unscaled - np.mean(y_test_unscaled),2) # L2 error norm
    if errors[iter,0] < best_error:
      best_error = errors[iter,0]
      best_model = model
  else:
    # Find most important sensors
    ig = captum.attr.IntegratedGradients(model)
    test_shap = torch.tensor(P_scaled_train[:,sensors])
    if use_GPU:
      test_shap = test_shap.to(device)
    test_shap.requires_grad_()
    attr = ig.attribute(test_shap,target=0,n_steps=50)
    attr = attr.cpu().detach().numpy()
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

"""# Testing and ploting"""
plt.rcParams['figure.figsize'] = [15, 7.5] # change figure size
# plt.rcParams['text.usetex'] = True
plt.rc('axes', labelsize=25, titlesize=25 )
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

# sensors = np.arange(n_probe)
# sensors = cluster_centers_indices
# test using the 29th sample from 26100th to 27000th
X_test = P_eval[:,sensors]
y_test = V_eval

with torch.no_grad():
  # input = P_val_input.unsqueeze(1)
  if use_GPU:
    X_test_temp = torch.tensor(X_test).to(device)
  else:
    X_test_temp = torch.tensor(X_test)
  test_output = model(X_test_temp).cpu().detach().numpy()

  if normalization == "01":
    test_output = unnormalize_01(test_output, V_min, V_max)
    y_test_unscaled = unnormalize_01(y_test, V_min, V_max)
  elif normalization == "m11":
    test_output = unnormalize_m1_1(test_output, V_mean, V_range)
    y_test_unscaled = unnormalize_m1_1(y_test, V_mean, V_range)
  elif normalization == "u_tau":
    test_output = unnormalize_utau(test_output, V_mean, utau, scale=1.0, square=False)
    y_test_unscaled = unnormalize_utau(y_test, V_mean, utau, scale=1.0, square=False)
  elif normalization == "u_tau_01":
    test_output = unnormalize_utau_01(test_output, V_mean, V_min, V_max, utau, scale=1.0, square=False)
    y_test_unscaled = unnormalize_utau_01(y_test, V_mean, V_min, V_max, utau, scale=1.0, square=False)
  elif normalization == "standard":
    test_output = scaler.inverse_transform(test_output.reshape(-1, 1)).reshape(-1)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

# compute L2 error
error2 = np.linalg.norm(test_output.reshape((-1)) - y_test_unscaled,2)/np.linalg.norm(y_test_unscaled - np.mean(y_test_unscaled),2) # L2 error norm

# compute correlation between test_output and y_test_unscaled
test_corr = np.corrcoef(test_output.reshape((-1)), y_test_unscaled)[0,1]

time_list = np.arange(test_output.shape[0])*dt_plus
plt.plot(time_list,test_output,color='b',label='Predicted')
plt.plot(time_list, y_test_unscaled,':',color='r',label='True')
# plt.legend(loc='upper right',fontsize=20)

error2_str = r'$\varepsilon =$ %1.3f' % error2
corr_str = r'$corr(v_{\mathrm{true}},v_{\mathrm{pred}}) =$ %1.3f' % test_corr
plt.annotate(error2_str,(20,50),xycoords='axes points',fontsize=30)
plt.annotate(corr_str,(20,20),xycoords='axes points',fontsize=28)

plt.xlabel('$t^+$')
plt.ylabel('$V(y^+=10)$')
plt.show()


model.train()  # Needed for IG

n_sensors_attr = 20
ig = captum.attr.IntegratedGradients(model)
test_shap = torch.tensor(P_scaled_train[:,sensors])
if use_GPU:
  test_shap = test_shap.to(device)
test_shap.requires_grad_()

# batch computation for memory saving
batch_size_attr = 50
attr_list = []

for i in range(0, P_scaled_train.shape[0], batch_size_attr):
    batch = test_shap[i:i+batch_size_attr]
    # batch.requires_grad_()
    attr_batch = ig.attribute(batch, target=0, n_steps=50)
    attr_list.append(attr_batch.cpu())
attr = torch.cat(attr_list, dim=0).detach().numpy()

# attr = ig.attribute(test_shap,target=0,n_steps=70)
# attr = attr.cpu().detach().numpy()
weights_mean = np.mean(abs(attr),axis=0)
sensor_temp = np.flip(np.argsort(weights_mean))
attr_sensors = sensors[sensor_temp[:n_sensors_attr]] # Indices of the attribution sensor locations
sensors_str = ','.join(map(str, attr_sensors)) #print "sensors" array as string and add commas
print(str(n_sensors_attr)+" attribution sensors:",sensors_str)

print(attr_sensors)

attr_sensors = np.array([182,187,129,139,106,243,231,204,93,260,283,267,226,103,134,300,218,294,63,197]) # 20 CAAF sensors
attr_sensors2 = np.array([0,   9,  18, 114, 123, 132, 228, 237, 246, 351])  # evenly spaced uniform sensors

fig = plt.figure(figsize=(10, 8))
# Create scatter plots for all points
plt.axvline(x=0, color='black', linestyle=':')
plt.axhline(y=0, color='black', linestyle=':')

# Add X-corr
data = scipy.io.loadmat('CrossCorr_mean.mat')
CrossCorr_mean = data['CrossCorr_mean']
x_plus_list = np.arange(-10,11)*dx
Xgrid, Zgrid = np.meshgrid(x_plus_list, x_plus_list)
contour = plt.contourf(Xgrid, Zgrid, CrossCorr_mean.T, levels=500, linestyles='none',cmap='viridis')
contour.set_edgecolor("face") # make sure there's no contours lines in AI

plt.colorbar(contour, ticks=[0, 0.5, 1])
plt.clim(0, 1)  # Set color limits

for j in attr_sensors2[:10]:
  plt.scatter(x_flatten[j], z_flatten[j], marker='o', s=100, facecolors='none', color='red', linewidth=2)

rank = 1
for i in attr_sensors[:10]:
  plt.scatter(x_flatten[i], z_flatten[i], marker='x', s=100, color='white', linewidth=2)
  annotation = plt.annotate(str(rank), (x_flatten[i]-1, z_flatten[i]+2), fontsize=22, color='white')
  rank += 1

plt.ylim([-50,50])
plt.yticks([-50,-25,0,25,50])
plt.xlim([-50,50])
plt.xticks([-50,-25,0,25,50])
plt.xlabel('$\Delta x^+$')
plt.ylabel('$\Delta z^+$')
# plt.gca().set_aspect('equal', adjustable='datalim')
plt.tick_params(axis='both', labelsize=22)
plt.savefig("PV_sensors.png",format='png',dpi=600)
plt.savefig("PV_sensors.pdf",format='pdf',bbox_inches='tight')
plt.show()

# Plot 2 models together
use_GPU = False
sensors_CAAF = np.array([182,187,129,139,106,243,231,204,93,260]) # 20 CAAF sensors with best pred
sensors_uniform = np.array([0,   9,  18, 114, 123, 132, 228, 237, 246, 351])  # evenly spaced
sensors_cluster = cluster_centers_indices

device = torch.device("cpu")
# load models trained using different sensor configurations
model_clustered = torch.load("Channel_PV_0882.pt",weights_only=False)
model_uniform = torch.load("Channel_PV_uniform.pt",weights_only=False)
model_CAAF = torch.load("Channel_PV_CAAF_0736.pt",weights_only=False)
model_clustered = model_clustered.to(device)
model_uniform = model_uniform.to(device)
model_CAAF = model_CAAF.to(device)

plt.rcParams['figure.figsize'] = [15, 7] # change figure size
plt.rc('axes', labelsize=25, titlesize=25 )
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)

X_test_CAAF = P_eval[:,sensors_CAAF]
X_test_uniform = P_eval[:,sensors_uniform]
X_test_clustered = P_eval[:,sensors_cluster]
y_test = V_eval

with torch.no_grad():
  # input = P_val_input.unsqueeze(1)
  X_test_temp = torch.tensor(X_test_clustered)
  test_output_clustered = model_clustered(X_test_temp).cpu().detach().numpy()
  X_test_temp = torch.tensor(X_test_CAAF)
  test_output_CAAF = model_CAAF(X_test_temp).cpu().detach().numpy()
  X_test_temp = torch.tensor(X_test_uniform)
  test_output_uniform = model_uniform(X_test_temp).cpu().detach().numpy()

  test_output_clustered = unnormalize_m1_1(test_output_clustered, V_mean, V_range)
  test_output_CAAF = unnormalize_m1_1(test_output_CAAF, V_mean, V_range)
  test_output_uniform = unnormalize_m1_1(test_output_uniform, V_mean, V_range)
  y_test_unscaled = unnormalize_m1_1(y_test, V_mean, V_range)

time_list = np.arange(P_eval.shape[0])*dt_plus-400
plt.plot(time_list,test_output_uniform/utau,color='r',label='Predicted  (uniform)')
plt.plot(time_list,test_output_CAAF/utau,color='b',label='Predicted  (CAAF)')
# plt.plot(time_list,test_output_clustered,color='g',label='Predicted  (clusters)')
plt.plot(time_list, y_test_unscaled/utau,':',color='black',label='True')


plt.xlabel(r'$t^+$')
plt.ylabel(r'$v_{10}^+$')
plt.ylim([-1,1])
plt.xlim([0,400])
plt.tick_params(axis='both', labelsize=20)
plt.tight_layout()

plt.savefig("PV_Pred_Compare.pdf",format='pdf')
plt.legend(loc='best',fontsize=20)
plt.show()