import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import seaborn as sns
# %pip install colorcet
import colorcet as cc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import *
import captum
import scipy.io
import h5py
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['text.usetex'] = True
plt.rc('axes', labelsize=20, titlesize=20 )
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)

"""# Data Setup"""

# Load Datasets
data = scipy.io.loadmat('tau_w.mat')
tau_w = data['tau_w']
with h5py.File('U_probe.mat', 'r') as file:
    U_probe_raw = file['U_probe'][()].T # Access the 'U_probe' dataset and convert to NumPy array

data = scipy.io.loadmat('X.mat')
X = data['x']
data = scipy.io.loadmat('Y.mat')
Y = data['ym']

print(tau_w.shape)
print(U_probe_raw.shape)
print(X.shape)
print(Y.shape)

n_points = U_probe_raw.shape[0]*U_probe_raw.shape[1]*U_probe_raw.shape[2]
print('n_points =',n_points)
n_xprobe = U_probe_raw.shape[3]
n_yprobe = U_probe_raw.shape[4]
n_probe = n_xprobe*n_yprobe

U_probe = U_probe_raw.reshape(n_points,n_xprobe,n_yprobe)
print(U_probe.shape)

# Exclude y+ < 10
yi_start = 22
yi_end = 42
U_probe = U_probe[:,:,yi_start:yi_end]
n_probe = U_probe.shape[1]*U_probe.shape[2]  # compute number probes after the removal
print(U_probe.shape)

dx = np.ediff1d(X)[0]
y_plus = Y
utau = 0.057231
Retau = 186
X_probe = X[99-10:99+11]-X[99]
Y_probe = Y[yi_start:yi_end]
x_grid, y_grid = np.meshgrid(X_probe, Y_probe)
x_flatten = x_grid.reshape(n_probe)*Retau
y_flatten = y_grid.reshape(n_probe)*Retau
U_flatten = U_probe.reshape(n_points,n_probe,order='F')
print('n_probe =',n_probe)

# Local normalization
U_max = np.amax(U_flatten,axis=0)
U_min = np.amin(U_flatten,axis=0)
U_scaled = (U_flatten-U_min)/(U_max-U_min)

tau_max = np.amax(tau_w)
tau_min = np.amin(tau_w)
tau_scaled = (tau_w-tau_min)/(tau_max-tau_min)
tau_scaled = tau_scaled.reshape((-1))
print(U_scaled.shape)
print(tau_scaled.shape)

fig = plt.figure(figsize=(8, 6))

# Create scatter plots for all points
plt.scatter(x_flatten, y_flatten, s=20, marker='o', c='black')
plt.axvline(x=0, color='black', linestyle='-')

plt.ylim([0,50])
plt.xlabel('$X^+$')
plt.ylabel('$Y^+$')
plt.title('Probe Locations')
#plt.savefig("clusters_map.png",format='png',bbox_inches='tight')
plt.show()

"""# Remove Correlated Sensors"""

# AffinityPropagation Clustering
aff_matrix = np.abs(np.corrcoef(U_flatten.T)) # Compute affinity matrix using correlation coefficient
af = AffinityPropagation(damping=0.5,max_iter=10000, convergence_iter=10, copy=True, preference= 0.95, affinity='precomputed').fit(aff_matrix)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(set(labels))
# sensors = cluster_centers_indices
print('n_clusters =',n_clusters_)

fig = plt.figure(figsize=(8, 6))
color_list = sns.color_palette(palette=cc.glasbey_category10,n_colors=n_clusters_)

# Create scatter plots for all points
plt.axvline(x=0, color='black', linestyle='--')
plt.scatter(x_flatten, y_flatten, s=20, marker='o', c=[color_list[i] for i in labels])
plt.scatter(x_flatten[cluster_centers_indices],y_flatten[cluster_centers_indices],marker='o', s=90, linewidths=2, color='black',facecolors='none')

plt.ylim([0,50])
plt.xlabel('$x^+$')
plt.ylabel('$y^+$')
# plt.title('Clusters')
plt.savefig("WSS_clusters.png",format='png',bbox_inches='tight')
plt.savefig("WSS_clusters.eps",format='eps')
plt.show()

np.save('cluster_centers_indices.npy', cluster_centers_indices)
np.save('labels.npy', labels)

cluster_centers_indices = np.load('cluster_centers_indices.npy')
labels = np.load('labels.npy')
n_clusters_ = len(set(labels))
print(n_clusters_)

fig = plt.figure(figsize=(8, 6))
color_list = sns.color_palette(palette=cc.glasbey_category10,n_colors=n_clusters_)

# Create scatter plots for all points
plt.axvline(x=0, color='black', linestyle='--')
plt.scatter(x_flatten, y_flatten, s=20, marker='o', c=[color_list[i] for i in labels])
plt.scatter(x_flatten[cluster_centers_indices],y_flatten[cluster_centers_indices],marker='o', s=90, linewidths=2, color='black',facecolors='none')

plt.ylim([0,45])
plt.xlabel('$X^+$')
plt.ylabel('$Y^+$')
# plt.title('Clusters')
#plt.savefig("clusters_map.png",format='png',bbox_inches='tight')
plt.show()

sensors = cluster_centers_indices

"""# Model Training and Attribution"""

# SET Testing Sensors HERE

n_iter = 3  # how many training runs to average
n_sensors = 5 # number of desired sensors
test_split = 0.1  # testing split
# sensors = np.array([102,145,37,40,93,183,98,225,34,29,153,250,156,219,23]) # 15 attr sensors for y+ > 10 BEST SO FAR!!!!!!

# sensors = np.array([4,7,10,13,16]) # 5 sensors at y = 10.09
sensors = cluster_centers_indices

mode = 'test'  # 'test' mode evaluates the performance of the given "sensors"
# mode = 'attr' # 'attr' mode performs attribution to rank the candidate sensors
#-------------------------------------------------------------------------------
best_model = None
best_error = 10.0
errors = np.zeros([n_iter,1])
weights_mean_all = np.zeros([n_iter,len(sensors)])
for iter in range(n_iter):
  print('Starting run', iter)
  random_state = np.random.randint(0, 201)
  X_train, X_test, y_train, y_test = train_test_split(
      U_scaled[:,sensors],
      tau_scaled,
      test_size=test_split,
      shuffle=False,
      random_state=random_state
  )

  # make datasets and dataloaders
  train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
  test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

  # set random seed
  torch.manual_seed(random_state)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)


  # Model for attribution with candidate sensors (>17 sensors)
  layer_size = 8
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

  optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-2)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-7, eps=1e-5, patience=3)
  loss_fn = nn.MSELoss()
  # n_epochs = 20
  threshold = 5e-5   # early stopping threshold for test loss
  n_epochs_min = 20
  n_epochs_max = 150
  min_loss = 10.0
  print_interval = 1

  # store metrics
  training_loss_history = np.zeros([n_epochs_max, 1])
  validation_loss_history = np.zeros([n_epochs_max, 1])

  for epoch in range(n_epochs_max):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
          target = target.unsqueeze(1)
          # Erase accumulated gradients
          optimizer.zero_grad()

          # Forward pass
          # data = torch.squeeze(data)
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
              target = target.unsqueeze(1)
              output = model(data)
              validation_loss_history[epoch] += loss_fn(output, target).item()  # Sum up batch loss

          validation_loss_history[epoch] /= len(test_loader)
          if epoch % print_interval == 0:
            print(f', Test Loss: %1.4e' % validation_loss_history[epoch,0],end='')

      # early stopping criteria
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


  if mode == 'test':
    with torch.no_grad():
      test_output = model(torch.tensor(X_test)).detach().numpy()

    # unscale test_output
    test_output = test_output*(tau_max - tau_min) + tau_min
    y_test_unscaled = y_test*(tau_max - tau_min) + tau_min

    # compute L2 error
    errors[iter,0] = np.linalg.norm(test_output.reshape((-1)) - y_test_unscaled,2)/np.linalg.norm(y_test_unscaled - np.mean(y_test_unscaled),2) # L2 error norm
    if errors[iter,0] < best_error:
      best_error = errors[iter,0]
      best_model = model
  else:
    # Find most important sensors
    ig = captum.attr.IntegratedGradients(model)
    test_shap = torch.tensor(U_scaled[sensors, :].T)
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

# Plotting--------------------------------------------------------
# attr_sensors = np.array([102,145,37,40,93]) # CAAF sensors
attr_sensors = np.array([4,7,10,13,16]) # 5 sensors at y = 10.09

fig = plt.figure(figsize=(8, 6))

# Create scatter plots for all points
plt.axvline(x=0, color='black', linestyle='--')
rank = 1
for i in attr_sensors[:5]:
  plt.scatter(x_flatten[i], y_flatten[i], marker='x', s=100, color='black', linewidth=2)
  annotation = plt.annotate(str(rank), (x_flatten[i]-1, y_flatten[i]+2), fontsize=18, color='black')
  rank += 1

plt.ylim([0,50])
plt.xlim([-60,60])
# plt.ylim([0,30])
plt.xlabel('$x^+$')
plt.ylabel('$y^+$')
plt.savefig("WSS_Uniform_5.png",format='png',bbox_inches='tight')
plt.savefig("WSS_Uniform_5.eps",format='eps')
plt.show()


# Test with time series data
plt.rcParams['figure.figsize'] = [15, 7.5] # change figure size
plt.rc('axes', labelsize=25, titlesize=25 )
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

with torch.no_grad():
  # input = P_val_input.unsqueeze(1)
  test_output = model(torch.tensor(X_test)).detach().numpy()

# unscale test_output
test_output = test_output*(tau_max - tau_min) + tau_min
y_test_unscaled = y_test*(tau_max - tau_min) + tau_min
# compute L2 error

error2 = np.linalg.norm(test_output.reshape((-1)) - y_test_unscaled,2)/np.linalg.norm(y_test_unscaled - np.mean(y_test_unscaled),2) # L2 error norm

n_skip = U_probe_raw.shape[1]*U_probe_raw.shape[2] # choose only one sample from each snapshot
plt.plot(test_output[0::n_skip],color='b',label='Predicted')
plt.plot(y_test_unscaled[0::n_skip],':',color='r',label='True')
# plt.legend(loc='upper right',fontsize=20)

error2_str = r'$\varepsilon =$ %1.3f' % error2
plt.annotate(error2_str,(20,50),xycoords='axes points',fontsize=30)

plt.xlabel('Time Index')
plt.ylabel('$\\tau_{w}$')
plt.show()