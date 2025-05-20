import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
from sklearn.model_selection import train_test_split
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import *
import captum
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rc('axes', labelsize=22, titlesize=22 )
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.rcParams['text.usetex'] = True

"""## Utilities"""

def create_dataset(x_in, y_in, lookback, pred_window):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
        pred_window: Size of window to predict (should be smaller than lookback)
    """
    n_output = len(x_in) - lookback - pred_window + 1
    # stop = len(x_in) - lookback + 1
    X, y = np.zeros((n_output,lookback,len(sensors))), np.zeros((n_output,lookback,1))
    for i in range(n_output):
        feature = x_in[i:i+lookback,:]
        # target = y_in[i+lookback:i+lookback+pred_window]
        target = y_in[i+pred_window:i+lookback+pred_window]
        X[i,:,:] = feature
        y[i,:,0] = target
    return X, y

"""# Data Setup"""

# SET PARAMETERS HERE
n_sensors = 10  # desired number of sensors
sensors = range(376)  # candidate sensor index

case_list = [0,1,2,3,4]   # validation case index
train_index = [0,1,2]   # training case index
nt_train = 30000  # number of time steps to be used for training
nt_val = 10000  # number of time steps to be used for testing and validation
n_augment = [0,0,2] # number of times to repeat the data for each training case for data augmentation
n_pred = 0  # number of future steps to predict the CL for (0 for current-time reconstruction)
n_P = 1 # number of history steps used to make the predictions
interval = 0.0025 # uniform time interval to interpolate the data onto

# "0" if normalized by sensors, "None" if normalized by case
NorBy = 0

# interpolation methods
P_inter = 'cubic'
CL_inter = 'cubic'

"""# load data for clustering"""

# Load Training Data
datasets = ["airfoil_5","airfoil_11","cylinder_5","cylinder_11","cylinder_0_5_1"]
n_grids = [188,188,188,208,208]
P_all = np.array([],ndmin=3)

for i in train_index:
  data1 = np.genfromtxt("Data/"+datasets[i]+"_wall1_surfacepressure_span.dat",delimiter = ',',usecols=range(n_grids[i]+1))

  df = pd.read_csv("Data/"+datasets[i]+"_CL.csv")
  CL_data = df.to_numpy()
  time_CL = CL_data[:,0]

  time_raw = time_CL[np.arange(3,len(data1[:,0])*4,4)] # get raw time data for pressure from CL data (per 4 entries)
  time_spaced = np.arange(time_raw[0],time_raw[-1],interval) # linear time space with interval
  print('Available data points =',time_spaced.size)

  time = time_spaced[:nt_train]

  # interpolate pressure to linear time space
  P1_raw = data1[:,1:]
  f = interpolate.interp1d(time_raw, P1_raw, kind='cubic', axis=0)
  P1_spaced = f(time_spaced)
  P_temp=P1_spaced[:nt_train,:]

  coord1 = np.genfromtxt("Data/"+datasets[2]+"_xy_sort_wall1.dat",delimiter = ',')
  nx = coord1[0,0].astype(int)
  coord1 = coord1[1:nx+1,:]
  X1=coord1[:,0] # X,Y coordinates in ascending order
  Y1=coord1[:,1]

  data2 = np.genfromtxt("Data/"+datasets[i]+"_wall2_surfacepressure_span.dat",delimiter = ',',usecols=range(n_grids[i]+1))
  # interpolate pressure to linear time space
  P2_raw = data2[:,1:]
  f = interpolate.interp1d(time_raw, P2_raw, kind='cubic', axis=0)
  P2_spaced = f(time_spaced)
  P2_temp=P2_spaced[:nt_train,:]

  coord2 = np.loadtxt("Data/"+datasets[2]+"_xy_sort_wall2.dat",skiprows=1)
  coord2 = coord2[0:nx,:]
  X2=coord2[:,0] # X,Y coordinates in ascending order
  Y2=coord2[:,1]

  # Interpolate sensor locations to the same spatial grid
  coord_temp = np.genfromtxt("Data/"+datasets[i]+"_xy_sort_wall1.dat",delimiter = ',')
  nx_temp = coord_temp[0,0].astype(int)
  coord_temp = coord_temp[1:nx_temp+1,:]
  X_temp = coord_temp[:,0]

  f = interpolate.interp1d(X_temp, P_temp, kind=P_inter, axis=1, fill_value='extrapolate')
  P1 = f(X1)

  f2 = interpolate.interp1d(X_temp, P2_temp, kind=P_inter, axis=1, fill_value='extrapolate')
  P2 = f2(X2)

  P_temp = (np.hstack((P1,P2)))[:,sensors]

  P_nor = (P_temp-np.mean(P_temp,axis=NorBy))/(np.amax(P_temp,axis=NorBy)-np.amin(P_temp,axis=NorBy)) # Normalization by case and sensor

  if not np.any(P_all): # if P_all is empty
    P_all = P_nor
  else:
    P_all = np.vstack((P_all,P_nor))

P_all = np.swapaxes(P_all,0,1) # P_all(n_samples, n_features)
print("P_all: ",np.shape(P_all))

# AffinityPropagation Clustering
aff_matrix = np.abs(np.corrcoef(P_all)) # Compute affinity matrix using correlation coefficient
af = AffinityPropagation(damping=0.5,max_iter=10000, convergence_iter=20, copy=True, preference= None, affinity='precomputed').fit(aff_matrix)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(set(labels))
# sensors = cluster_centers_indices
print(n_clusters_)

# Make cluster centers the new candidate sensors
sensors = cluster_centers_indices

"""## (Optional) Other clustering methods"""

# import itertools

# # find cluster center using the highest correlated point
# cluster_centers_indices = np.zeros(n_clusters_,dtype=int)
# for i in range(n_clusters_):
#   cluster_sensor_ind = np.where(labels==i)[0]
#   combs = list(itertools.combinations(cluster_sensor_ind, 2))
#   corr_sum = np.zeros(len(cluster_sensor_ind))
#   for ind in combs:
#     corr_sum[np.where(cluster_sensor_ind==ind[0])[0][0]] += aff_matrix[ind[0],ind[1]]
#     corr_sum[np.where(cluster_sensor_ind==ind[1])[0][0]] += aff_matrix[ind[0],ind[1]]
#   cluster_centers_indices[i] = cluster_sensor_ind[np.argmax(corr_sum)]

# print(cluster_centers_indices)

# # DBSCAN Clustering
# # aff_matrix = 1-np.abs(np.corrcoef(P_all)) # Compute affinity matrix using correlation coefficient
# af = DBSCAN(eps=1.0, min_samples = 10, metric='correlation').fit(P_all)
# labels = af.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters_)

# # HDBSCAN Clustering
# # aff_matrix = 1-np.abs(np.corrcoef(P_all)) # Compute affinity matrix using correlation coefficient
# af = HDBSCAN(min_cluster_size=10, max_cluster_size=50, store_centers='centroid', copy=True, metric='correlation').fit(P_all)
# labels = af.labels_
# cluster_centers_indices = af.centroids_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters_)

# # OPTICS Clustering
# # aff_matrix = 1-np.abs(np.corrcoef(P_all)) # Compute affinity matrix using correlation coefficient
# af = OPTICS(max_eps=0.1, min_samples = 5, metric='correlation', xi = 0.05).fit(P_all)
# labels = af.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters_)

# # Spectral Clustering
# aff_matrix = 1-np.abs(np.corrcoef(P_all)) # Compute affinity matrix using correlation coefficient
# af = SpectralClustering(n_clusters=20, affinity='precomputed').fit(aff_matrix)
# labels = af.labels_
# n_clusters_ = len(set(labels))
# print(n_clusters_)

"""## Cluster visualization"""

# Project "P_all" onto a 3D field and visualize all 376 features

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

color_list = sns.color_palette(palette=cc.glasbey_category10,n_colors=n_clusters_)

# Perform PCA to reduce dimensionality to 3
pca = PCA(n_components=3)
P_all_pca = pca.fit_transform(P_all)

# Create a 3D plot
fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(111, projection='3d')

# Plot each feature as a point in 3D space
for i in range(P_all_pca.shape[0]):
    ax.scatter(P_all_pca[i, 0], P_all_pca[i, 1], P_all_pca[i, 2], marker='o', color=color_list[labels[i]])

# Add labels and title
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
# ax.set_title('Projection of pressure onto 3D Coordiantes')
plt.show()

# Plot cluster results on the airfoil
color_list = sns.color_palette(palette=cc.glasbey_category10,n_colors=n_clusters_)
point_color = [color_list[i] for i in labels]
plt.rcParams['figure.figsize'] = [15, 3] # change figure size
plt.scatter(X1,Y1,marker='o', s=20,color=point_color[:len(X1)])
plt.scatter(X2,Y2,marker='o', s=20,color=point_color[len(X1):])
X_all = np.hstack((X1,X2))
Y_all = np.hstack((Y1,Y2))
plt.scatter(X_all[cluster_centers_indices],Y_all[cluster_centers_indices],marker='o', s=60, linewidths=2, color='black',facecolors='none')
plt.ylabel('$y/C$')
plt.ylim([-0.1,0.1])
plt.xlabel('$x/C$')
plt.tight_layout()
# plt.savefig('Figs/airfoil_clusters.png')
# plt.savefig('Figs/airfoil_clusters.eps',format = 'eps')
plt.show()

"""# Model Training and Attribution"""

n_iter = 10  # how many training runs to average
# mode = 'test'  # 'test' mode evaluates the performance of the given "sensors"
mode = 'attr' # 'attr' mode performs attribution to rank the candidate sensors
n_sensors = 10 # number of desired sensors
sensors = range(1,377) # use every surface node as candidate sensors
# sensors = np.array([232,257,164,280,63,17,183,191,0,139])+1 # 10 IG sensors using Affinity Propogation clustering and 15 iters of training
sensors = np.array(sensors)-1 # turn into 0-indexing

#-----------------------------------------------------------------------------------
# Load Validation Data
# 1. Load Pressure
datasets = ["airfoil_5","airfoil_11","cylinder_5","cylinder_11","cylinder_0_5_1"]
n_grids = [188,188,188,208,208]
n_val = nt_val-n_P-n_pred+1

CL_test = np.array([])
P_val = np.zeros((n_val,n_P,len(sensors),len(case_list)))
CL_val = np.zeros((n_val,n_P,1,len(case_list)))
time_val = np.zeros((n_val,len(case_list)))

for case_i in range(len(case_list)):
  case_index = int(case_list[case_i])
  data1 = np.genfromtxt("Data/"+datasets[case_index]+"_wall1_surfacepressure_span.dat",delimiter = ',',usecols=range(n_grids[case_index]+1))

  df = pd.read_csv("Data/"+datasets[case_index]+"_CL.csv")
  CL_data = df.to_numpy()
  time_CL = CL_data[:,0]

  time_raw = time_CL[np.arange(3,len(data1[:,0])*4,4)] # get raw time data for pressure from CL data (per 4 entries)
  time_spaced = np.arange(time_raw[0],time_raw[-1],interval) # linear time space with interval
  print('Available data points for case %i =' % (case_i),time_spaced.size)
  time_val[:,case_i] = time_spaced[-n_val:]

 # interpolate pressure to linear time space
  P1_raw = data1[:,1:]
  f = interpolate.interp1d(time_raw, P1_raw, kind='cubic', axis=0)
  P1_spaced = f(time_spaced)
  P1=P1_spaced[-nt_val:,:]

  coord1 = np.genfromtxt("Data/"+datasets[2]+"_xy_sort_wall1.dat",delimiter = ',')
  nx = coord1[0,0].astype(int)
  coord1 = coord1[1:nx+1,:]
  X1=coord1[:,0] # X,Y coordinates in ascending order
  Y1=coord1[:,1]

  data2 = np.loadtxt("Data/"+datasets[case_index]+"_wall2_surfacepressure_span.dat",delimiter = ',',usecols=range(n_grids[case_index]+1))

  # interpolate pressure to linear time space
  P2_raw = data2[:,1:]
  f = interpolate.interp1d(time_raw, P2_raw, kind='cubic', axis=0)
  P2_spaced = f(time_spaced)
  P2=P2_spaced[-nt_val:,:]

  coord2 = np.loadtxt("Data/"+datasets[2]+"_xy_sort_wall2.dat",skiprows=1)
  coord2 = coord2[0:nx,:]
  X2=coord2[:,0] # X,Y coordinates in ascending order
  Y2=coord2[:,1]

  # Interpolate validation pressure
  coord_temp = np.genfromtxt("Data/"+datasets[case_index]+"_xy_sort_wall1.dat",delimiter = ',')
  nx_temp = coord_temp[0,0].astype(int)
  coord_temp = coord_temp[1:nx_temp+1,:]
  X_temp = coord_temp[:,0]
  f = interpolate.interp1d(X_temp, P1, kind=P_inter, axis=1, fill_value='extrapolate')

  P1 = f(X1)
  f2 = interpolate.interp1d(X_temp, P2, kind=P_inter, axis=1, fill_value='extrapolate')
  P2 = f2(X2)

  P_temp = (np.hstack((P1,P2)))[:,sensors]
  P_temp = (P_temp-np.mean(P_temp,axis=NorBy))/(np.amax(P_temp,axis=NorBy)-np.amin(P_temp,axis=NorBy)) # Normalization by case & sensors

  CL_pre = CL_data[:,1]
  CL_interp= interpolate.interp1d(time_CL, CL_pre, kind=CL_inter, fill_value='extrapolate')
  CL_aft = CL_interp(time_spaced[-nt_val:])

  P_val_stacked, CL_val_stacked = create_dataset(P_temp, CL_aft, lookback=n_P, pred_window = n_pred)

  P_val[:,:,:,case_i] = P_val_stacked
  CL_val[:,:,:,case_i] = CL_val_stacked[:,-n_P:,0].reshape(-1,n_P,1)

  CL_val_norm = (CL_val_stacked-np.mean(CL_val_stacked))/(np.amax(CL_val_stacked)-np.amin(CL_val_stacked)) # normalize validation CL to use in testing
  if not np.any(CL_test): # if CL_test is empty
    CL_test = CL_val_norm
    P_test = P_val_stacked
  else:
    CL_test = np.vstack((CL_test,CL_val_norm))
    P_test = np.vstack((P_test,P_val_stacked))

  print("CL_test: ",np.shape(CL_test))

# Load Training Data
P_all = np.array([],ndmin=3)
CL = np.array([],ndmin=3)
for i in train_index:
  data1 = np.genfromtxt("Data/"+datasets[i]+"_wall1_surfacepressure_span.dat",delimiter = ',',usecols=range(n_grids[i]+1))

  df = pd.read_csv("Data/"+datasets[i]+"_CL.csv")
  CL_data = df.to_numpy()
  time_CL = CL_data[:,0]

  time_raw = time_CL[np.arange(3,len(data1[:,0])*4,4)] # get raw time data for pressure from CL data (per 4 entries)
  time_spaced = np.arange(time_raw[0],time_raw[-1],interval) # linear time space with interval
  print('Available data points for case %i =' % (i),time_spaced.size)

  time = time_spaced[:nt_train]

  # interpolate pressure to linear time space
  P1_raw = data1[:,1:]
  f = interpolate.interp1d(time_raw, P1_raw, kind='cubic', axis=0)
  P1_spaced = f(time_spaced)
  P_temp=P1_spaced[:nt_train,:]

  data2 = np.genfromtxt("Data/"+datasets[i]+"_wall2_surfacepressure_span.dat",delimiter = ',',usecols=range(n_grids[i]+1))
  # interpolate pressure to linear time space
  P2_raw = data2[:,1:]
  f = interpolate.interp1d(time_raw, P2_raw, kind='cubic', axis=0)
  P2_spaced = f(time_spaced)
  P2_temp=P2_spaced[:nt_train,:]

  CL_pre = CL_data[:,1]
  CL_interp= interpolate.interp1d(time_CL, CL_pre, kind=CL_inter, fill_value='extrapolate')
  CL_temp = CL_interp(time)

  CL_nor = (CL_temp-np.mean(CL_temp))/(np.amax(CL_temp)-np.amin(CL_temp)) # Normalization by case

  coord_temp = np.genfromtxt("Data/"+datasets[i]+"_xy_sort_wall1.dat",delimiter = ',')
  nx_temp = coord_temp[0,0].astype(int)
  coord_temp = coord_temp[1:nx_temp+1,:]
  X_temp = coord_temp[:,0]

  f = interpolate.interp1d(X_temp, P_temp, kind=P_inter, axis=1, fill_value='extrapolate')
  P1 = f(X1)

  f2 = interpolate.interp1d(X_temp, P2_temp, kind=P_inter, axis=1, fill_value='extrapolate')
  P2 = f2(X2)

  P_temp = (np.hstack((P1,P2)))[:,sensors]

  P_nor = (P_temp-np.mean(P_temp,axis=NorBy))/(np.amax(P_temp,axis=NorBy)-np.amin(P_temp,axis=NorBy)) # Normalization by case

  P_train_stacked, CL_train_stacked = create_dataset(P_nor, CL_nor, lookback=n_P, pred_window = n_pred)

  if not np.any(P_all): # if P_all is empty
    P_all = P_train_stacked
    CL = CL_train_stacked
  else:
    P_all = np.vstack((P_all,P_train_stacked))
    CL = np.vstack((CL,CL_train_stacked))

  for k in range(n_augment[i]):
    P_all = np.vstack((P_all,P_train_stacked))
    CL = np.vstack((CL,CL_train_stacked))

train_y = torch.squeeze(torch.tensor(CL), dim=2)
test_y = torch.squeeze(torch.tensor(CL_test), dim=2)

train_X = torch.tensor(P_all)
test_X = torch.tensor(P_test)

# make datasets and dataloaders
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
#-----------------------------------------------------------------------------------
errors = np.zeros([n_iter,len(case_list)])
weights_mean_all = np.zeros([n_iter,len(sensors)])
for iter in range(n_iter):
  print('Starting run', iter)
  random_state = np.random.randint(0, 201)
  # set random seed
  torch.manual_seed(random_state)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

  # Model for attribution with candidate sensors (>17 sensors)
  layer_size = 8
  model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(len(sensors)*n_P, layer_size,bias=True),
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

  # Model for testing sensor configurations
  if len(sensors) < 17:
    layer_size = 4
    model = nn.Sequential(
          nn.Flatten(),
          nn.Linear(len(sensors)*n_P, layer_size,bias=True),
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

  # Model for training using all 376 sensors
  if len(sensors) == 376:
    layer_size = 16
    model = nn.Sequential(
          nn.Flatten(),
          nn.Linear(len(sensors)*n_P, layer_size,bias=True),
          nn.BatchNorm1d(layer_size),
          nn.LeakyReLU(),
          nn.Linear(layer_size, layer_size,bias=True),
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

  optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-7, eps=1e-5, patience=1)
  loss_fn = nn.MSELoss()
  # n_epochs = 20
  threshold = 1e-5   # early stopping threshold for test loss
  n_epochs_min = 10
  n_epochs_max = 100

  # store metrics
  training_loss_history = np.zeros([n_epochs_max, 1])
  validation_loss_history = np.zeros([n_epochs_max, 1])

  for epoch in range(n_epochs_max):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
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
      print(f'Train Epoch: %d/%i  Train Loss: %1.4e' % (epoch + 1, n_epochs_max, training_loss_history[epoch,0]),end='')


      # Turning off automatic differentiation
      with torch.no_grad():
          model.eval()
          for data, target in test_loader:
              output = model(data)
              validation_loss_history[epoch] += loss_fn(output, target).item()  # Sum up batch loss

          validation_loss_history[epoch] /= len(test_loader)
          print(f', Test Loss: %1.4e' % validation_loss_history[epoch,0],end='')

      loss_rate = (training_loss_history[epoch-1]-training_loss_history[epoch])/training_loss_history[epoch]
      if epoch > n_epochs_min and loss_rate < threshold: # early stopping criteria
        break

      scheduler.step(validation_loss_history[epoch,0]) # adjust learning rate
      print(f', Learning Rate = %1.2e' % optimizer.param_groups[0]['lr'])

  # Plot loss history
  plt.rcParams['figure.figsize'] = [6, 5] # change figure size
  plt.plot(validation_loss_history[:epoch], color='b',linewidth=2, label='Test Loss')
  plt.plot(training_loss_history[:epoch],':',color='r',linewidth=2, label='Train Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

  # Compute and plot predictions for CL
  LiftRange =[(0,0.5), (0,2), (-4,4), (-4,4), (-4,4)] # Y range of lift plot
  # LiftRange =[(-4,4), (-4,4), (-4,4), (-4,4), (-4,4)] # Y range of lift plot
  datasets = ["airfoil_5","airfoil_11","cylinder_5","cylinder_11","cylinder_0_5_1"]
  case_names = ["None-5","None-11","Cylinder-5","Cylinder-11","Cylinder2-0"]

  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

  plt.rcParams['text.usetex'] = True
  plt.rc('axes', labelsize=25, titlesize=25 )
  plt.rc('xtick',labelsize=20)
  plt.rc('ytick',labelsize=20)

  fig, axs = plt.subplots(1,len(case_list),figsize=(len(case_list)*5, 4))
  if len(case_list) == 1:
    axs.set(ylabel='$C_{L}$')
  else:
    axs[0].set(ylabel='$C_{L}$')

  for case_i in range(len(case_list)):

    P_val_input = torch.tensor((P_val[:,:,:,case_i]))

    with torch.no_grad():
      # input = P_val_input.unsqueeze(1)
      val_output = model(P_val_input)

    CL_i = CL_val[:,:,:,case_i].reshape(-1)
    CL_pred = (val_output[:, -1].numpy())*(np.amax(CL_i)-np.amin(CL_i))+np.mean(CL_i) # Un-normalize predicted CL
    CL_val_i = CL_val[:,-1,0,case_i]
    if len(case_list) == 1:
      axs.plot(time_val[:,case_i]-time_val[0,case_i],CL_pred,color='b',linewidth=2, label='Predicted CL')
      axs.plot(time_val[:,case_i]-time_val[0,case_i],CL_val_i,':',color='r',linewidth=2, label='Simulated CL')
      axs.set(xlabel='$tU_{\infty}/C$',ylim=LiftRange[case_list[case_i]])
      axs.set_title(r"$\textbf{"+case_names[case_list[case_i]]+"}$")
    else:
      axs[case_i].plot(time_val[:,case_i]-time_val[0,case_i],CL_pred,color='b',linewidth=2, label='Predicted CL')
      axs[case_i].plot(time_val[:,case_i]-time_val[0,case_i],CL_val_i,':',color='r',linewidth=2, label='Simulated CL')
      axs[case_i].set(title=case_names[case_list[case_i]],xlabel='$tU_{\infty}/C$',ylim=LiftRange[case_list[case_i]],yticks=[-4,-2,0,2,4])
      axs[case_i].set(xlabel='$tU_{\infty}/C$',ylim=LiftRange[case_list[case_i]])
      axs[case_i].set_title(r"$\textbf{"+case_names[case_list[case_i]]+"}$")

    # n_pred = 1
    CL_pred = (val_output[:, -n_pred:].numpy())*(np.amax(CL_i)-np.amin(CL_i))+np.mean(CL_i)
    CL_val_i = CL_val[:,-n_pred:,0,case_i]
    error2 = np.linalg.norm(CL_val_i.reshape((-1)) - CL_pred.reshape((-1)),2)/np.linalg.norm(CL_val_i - np.mean(CL_val_i),2) # compute L2 error norm
    error2_str = r'$\varepsilon =$ %1.3f' % error2
    if len(case_list) == 1:
      axs.annotate(error2_str,(20,20),xycoords='axes points',fontsize=20)
    else:
      axs[case_i].annotate(error2_str,(20,20),xycoords='axes points',fontsize=20)
    errors[iter,case_i] = error2
  plt.show()

  if mode == 'attr':
    ig = captum.attr.IntegratedGradients(model)
    n_per_case = 10000 # how many data points from each case are used to evaluate the attribution

    # Sample data points from the training dataset for attribution
    P_test_shap = np.zeros((n_per_case*len(train_index),n_P,len(sensors)))
    for i in range(len(train_index)):
      if n_augment[i] > 0 and i < 2: print("Warning: data augmentation is applied and may cause incorrect sampling for attribution!")
      P_test_shap[i*n_per_case:(i+1)*n_per_case,:,:] = P_all[np.random.choice(np.arange(i*nt_train,(i+1)*nt_train), n_per_case, replace=False),:,:]  # randomly choose n_per_case time steps from the training data for attribution

    P_test_shap = torch.tensor(P_test_shap)
    P_test_shap.requires_grad_()

    # perform attribution
    attr = ig.attribute(P_test_shap,target=0,n_steps=15)
    attr = attr.detach().numpy()

    attr_avg_n_P = np.mean(abs(attr),axis=1) # average over pressure history steps (2nd axis)
    weights_mean_all[iter,:] = np.mean(attr_avg_n_P,axis=0) # average over time samples (1st axis)

# Sort and print optimal sensor indices
if mode == 'attr':
  weights_mean = np.mean(weights_mean_all,axis=0) # average attribution vlaues over all training runs
  sensor_temp = np.flip(np.argsort(weights_mean)) # rank sensors
  attr_sensors = sensors[sensor_temp[:n_sensors]] # get indices of the attribution sensor locations
  sensors_str = ','.join(map(str, attr_sensors)) # print "sensors" array as string and add commas
  print(str(n_sensors)+" attribution sensors:",sensors_str)
else:
  mean_errors = np.mean(errors,axis=0)
  print('mean errors =',','.join(map(str, mean_errors)))

## (optional) remove poorly trained iterations
# errors_cleaned = np.delete(errors,[2,7],0)
# print(errors_cleaned.shape)
# mean_errors = np.mean(errors_cleaned,axis=0)
print('mean errors =',','.join(map(str, np.round(mean_errors,3))))