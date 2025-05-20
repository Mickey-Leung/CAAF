import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import *
import captum

"""## Utilities"""

def generate_data_with_correlation(X, target_corr, random_seed=None):
    """
    Generate a new dataset Y with a specific correlation coefficient with X.

    Parameters:
        X (array-like): Original dataset.
        target_corr (float): Desired correlation coefficient (-1 to 1).
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        Y (numpy.ndarray): Generated dataset with the specified correlation coefficient.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Standardize X
    X = np.array(X)
    X_std = (X - np.mean(X)) / np.std(X)

    # Generate random Y
    Y_random = np.random.randn(len(X))
    Y_random_std = (Y_random - np.mean(Y_random)) / np.std(Y_random)

    # Create Y with the desired correlation
    Y = target_corr * X_std + np.sqrt(1 - target_corr**2) * Y_random_std

    # Scale Y back to match X's mean and standard deviation if needed
    Y = Y * np.std(X) + np.mean(X)

    return Y

def generate_data_with_correlation_2d(X, R, random_seed=None):
    """
    Generate a dataset Y with specific correlation coefficients with multiple given datasets.

    Parameters:
        X (numpy.ndarray): A 2D array of shape (n_datasets, n_datapoints), where each row is a dataset.
        R (array-like): Desired correlation coefficients for each dataset in X (1D array of length n_datasets).
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        Y (numpy.ndarray): Generated dataset with the desired correlation coefficients with each row of X.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Ensure X is a numpy array and has the correct shape
    X = np.asarray(X)
    n_datasets, n_datapoints = X.shape

    if len(R) != n_datasets:
        raise ValueError("The length of R must match the number of rows in X.")

    # Standardize each row in X to have zero mean and unit variance
    X_std = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    # Generate a random component orthogonal to all rows in X
    random_component = np.random.randn(n_datapoints)
    random_component = (random_component - np.mean(random_component)) / np.std(random_component)

    for i in range(n_datasets):
        # Orthogonalize the random component with respect to each row of X
        random_component -= np.dot(random_component, X_std[i]) * X_std[i] / np.dot(X_std[i], X_std[i])

    # Normalize the random component
    random_component = random_component / np.std(random_component)

    # Combine the standardized datasets in X and the random component to create Y
    Y = np.zeros(n_datapoints)
    for i in range(n_datasets):
        Y += R[i] * X_std[i]

    if np.sum(np.array(R)**2) < 1:
      Y += np.sqrt(1 - np.sum(np.array(R)**2)) * random_component
    else:
      print('Magnitude > 1. Random noise not added.')

    return Y


"""# Create Generated Dataset"""

# Demo 1: choosing 2 out of 3 sensors
# CL is the prediction target
# P1, P2, and P3 are the input features

random_seed=35
N = 10000 # number of datapoints
np.random.seed(None)
P1 = np.random.rand(N)
P2 = generate_data_with_correlation(P1, 0.9, random_seed=random_seed)
print('corr(P1,P2) =',np.round(np.corrcoef(P1,P2)[0,1],3))

np.random.seed(None)
P3 = np.random.rand(N)
print('corr(P1,P3) =',np.round(np.corrcoef(P1,P3)[0,1],3))

P_cluster = np.vstack((P1,P2,P3))
print(P_cluster.shape)

CL = generate_data_with_correlation_2d(P_cluster, [0.7, 0.2, 0.3], random_seed=random_seed)
print('corr(CL, P1) =',np.round(np.corrcoef(CL,P1)[0,1],3))
print('corr(CL, P2) =',np.round(np.corrcoef(CL,P2)[0,1],3))
print('corr(CL, P3) =',np.round(np.corrcoef(CL,P3)[0,1],3))
print('Average CL =',np.mean(CL))
CL = np.expand_dims(CL,axis=(1,2))
print(CL.shape)

# AffinityPropagation Clustering
aff_matrix = np.abs(np.corrcoef(P_cluster)) # Compute affinity matrix using correlation coefficient
af = AffinityPropagation(damping=0.5,max_iter=10000, convergence_iter=10, copy=True, preference= 0.7, affinity='precomputed').fit(aff_matrix)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(set(labels))
sensors = cluster_centers_indices
print(f'n_clusters = {n_clusters_}')
print(f'labels = {labels}')
print(f'cluster centers = {cluster_centers_indices}')

sensors = cluster_centers_indices  # set the cluster centers as candidates
sensors = range(3)  # set all 3 sensors as candidates

"""###Other Demos"""

# Method demo2: choosing 3 out of 5 sensors

random_seed=33
N = 10000 # number of datapoints
np.random.seed(None)
P1 = np.random.rand(N)
P2 = generate_data_with_correlation(P1, 0.7, random_seed=random_seed)
print('corr(P1,P2) =',np.round(np.corrcoef(P1,P2)[0,1],3))
P3 = generate_data_with_correlation(P1, 0.5, random_seed=random_seed)
print('corr(P1,P3) =',np.round(np.corrcoef(P1,P3)[0,1],3))
print('corr(P2,P3) =',np.round(np.corrcoef(P2,P3)[0,1],3))
np.random.seed(None)
P4 = np.random.rand(N)
print('corr(P1,P4) =',np.round(np.corrcoef(P1,P4)[0,1],3))
np.random.seed(None)
P5 = np.random.rand(len(P4))
print('corr(P1,P5) =',np.round(np.corrcoef(P4,P5)[0,1],3))

P_cluster = np.vstack((P1,P2,P3,P4,P5))
print(P_cluster.shape)

CL = generate_data_with_correlation_2d(P_cluster, [0.5, 0.3, 0.3, 0.4, 0.0], random_seed=random_seed)
# print CL's correlation with every row in P_cluster
print('corr(CL, P1) =',np.round(np.corrcoef(CL,P1)[0,1],3))
print('corr(CL, P2) =',np.round(np.corrcoef(CL,P2)[0,1],3))
print('corr(CL, P3) =',np.round(np.corrcoef(CL,P3)[0,1],3))
print('corr(CL, P4) =',np.round(np.corrcoef(CL,P4)[0,1],3))
print('corr(CL, P5) =',np.round(np.corrcoef(CL,P5)[0,1],3))
print('Average CL =',np.mean(CL))
CL = np.expand_dims(CL,axis=(1,2))
print(CL.shape)

# Extreme case 1

random_seed=50
N = 10000 # number of datapoints
np.random.seed(random_seed)
P1 = np.random.rand(N)
P_cluster = P1.reshape(1,-1)
print(P_cluster.shape)
CL = generate_data_with_correlation(P1, 0.8, random_seed=random_seed)
print('corr(CL, P1) =',np.round(np.corrcoef(CL,P1)[0,1],3))
CL = np.expand_dims(CL,axis=(1,2))
print(CL.shape)
sensors = range(P_cluster.shape[0])

# Extreme case 2

random_seed=50
N = 10000 # number of datapoints
np.random.seed(None)
P1 = np.random.rand(N)
P2 = np.random.rand(N)
print('corr(P1,P2) =',np.round(np.corrcoef(P1,P2)[0,1],3))
P3 = np.random.rand(N)
print('corr(P1,P3) =',np.round(np.corrcoef(P1,P3)[0,1],3))
P_cluster = np.vstack((P1,P2,P3))
print(P_cluster.shape)

CL = generate_data_with_correlation(P1, 1.0, random_seed=random_seed)

# print CL's correlation with every row in P_cluster
print('corr(CL, P1) =',np.round(np.corrcoef(CL,P1)[0,1],3))
print('corr(CL, P2) =',np.round(np.corrcoef(CL,P2)[0,1],3))
print('corr(CL, P3) =',np.round(np.corrcoef(CL,P3)[0,1],3))
CL = np.expand_dims(CL,axis=(1,2))
print(CL.shape)

sensors = range(3)

"""# Model Training and Attribution"""

n_iter = 1
num_train = int(0.9 * len(CL)) # how many data points reserved for training

P_temp = np.expand_dims(P_cluster.T,axis=1)
P_input = P_temp[:,:,sensors]

train_y = torch.squeeze(torch.tensor(CL[:num_train,:,:]), dim=2)
test_y = torch.squeeze(torch.tensor(CL[num_train:,:,:]), dim=2)

train_X = torch.tensor(P_input[:num_train,:,:])
test_X = torch.tensor(P_input[num_train:,:,:])

# make datasets and dataloaders
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)

weights_mean_all = np.zeros([n_iter,len(sensors)])
for iter in range(n_iter):
  print('Starting run', iter)
  random_state = np.random.randint(0, 201)
  # set random seed
  torch.manual_seed(random_state)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

  # Model for testing sensor configurations
  layer_size = 4
  model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(len(sensors), layer_size,bias=True),
        nn.LeakyReLU(),
        nn.Linear(layer_size, layer_size,bias=True),
        nn.LeakyReLU(),
        nn.Linear(layer_size, layer_size,bias=True),
        nn.LeakyReLU(),
        nn.Linear(layer_size, 1,bias=True),
  )

  model = model.to(torch.double)
  print(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-7, eps=1e-8, patience=1)
  loss_fn = nn.MSELoss()
  threshold = 1e-12   # early stopping threshold for test loss
  n_epochs_min = 30
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
      if epoch % 10 == 0:
        print(f'Train Epoch: %d/%i  Train Loss: %1.4e' % (epoch + 1, n_epochs_max, training_loss_history[epoch,0]),end='')


      # Turning off automatic differentiation
      with torch.no_grad():
          model.eval()
          for data, target in test_loader:
              output = model(data)
              validation_loss_history[epoch] += loss_fn(output, target).item()  # Sum up batch loss

          validation_loss_history[epoch] /= len(test_loader)
          if epoch % 10 == 0:
            print(f', Test Loss: %1.4e' % validation_loss_history[epoch,0],end='')

      # loss_rate = (validation_loss_history[epoch-1]-validation_loss_history[epoch])/validation_loss_history[epoch]
      loss_rate = (training_loss_history[epoch-1]-training_loss_history[epoch])/training_loss_history[epoch]
      if epoch > n_epochs_min and loss_rate < threshold: #loss_rate > 0 and
        break

      scheduler.step(validation_loss_history[epoch,0]) # adjust learning rate
      if epoch % 10 == 0:
        print(f', Learning Rate = %1.2e' % optimizer.param_groups[0]['lr'])

  # Plot loss history
  plt.rcParams['figure.figsize'] = [6, 5] # change figure size
  plt.plot(validation_loss_history[:epoch], color='b',linewidth=2, label='Test Loss')
  plt.plot(training_loss_history[:epoch],':',color='r',linewidth=2, label='Train Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

  # Compute attribution value by inputting validation data
  ig = captum.attr.IntegratedGradients(model)

  P_test_shap = torch.tensor(P_input)
  P_test_shap.requires_grad_()

  attr = ig.attribute(P_test_shap,target=0,n_steps=50)
  attr = attr.detach().numpy()

  # Compute mean attribution value for each candidate sensor
  attr_avg_n_P = np.mean((attr),axis=1)  # average over pressure history steps (2nd axis)
  weights_mean_all[iter,:] = np.mean(attr_avg_n_P,axis=0) # average over time samples (1st axis)

weights_mean = np.mean(weights_mean_all,axis=0) # average over iterations
sensor_temp = np.flip(np.argsort(weights_mean))
print(sensor_temp) # Print the sensor ranking

print('Absolute attributions:',np.round(weights_mean,2))
print('Total attribution:',np.round(np.sum(weights_mean),2))
print('Percent attribution:',np.round(abs(weights_mean)/np.sum(abs(weights_mean))*100,0))