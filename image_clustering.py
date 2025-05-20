
import torch
import numpy as np
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
from sklearn.cluster import *
from PIL import Image
import os
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rc('axes', labelsize=20, titlesize=20 )
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)

%cd .../ImageNet        # make sure current directory contains the ImageNet data

"""## Utilities"""

def load_jpeg_images(directory):
    """Loads JPEG images from a directory into a list of PIL Image objects.

    Args:
        directory: The path to the directory containing the JPEG images.

    Returns:
        A list of PIL Image objects, or None if an error occurs.
    """
    image_list = []
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
                filepath = os.path.join(directory, filename)
                try:
                    img = Image.open(filepath)
                    image_list.append(img)
                except IOError:
                    print(f"Error opening image file: {filepath}")
                    # Handle the error appropriately (e.g., skip the file or raise an exception)

    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return None

    print(f"Loaded {len(image_list)} images.")
    return image_list

def image_to_3d_array(image):
    """Converts a PIL Image object to a 3D NumPy array.

    Args:
        image: A PIL Image object.

    Returns:
        A 3D NumPy array representing the image, or None if an error occurs.
    """
    try:
        # Convert the image to RGB mode if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert the image to a NumPy array
        image_array = np.array(image)/255

        return image_array
    except Exception as e:
        print(f"Error converting image to array: {e}")
        return None

# downsample with average pooling for image_array
def downsample_image(image_array, kernel_size=2, stride=2):
    """Downsamples an image using average pooling.

    Args:
        image_array: A NumPy array representing the image.
        kernel_size: The size of the pooling kernel.
        stride: The stride of the pooling operation.

    Returns:
        A NumPy array representing the downsampled image.
    """

    # Convert the image array to a PyTorch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Apply average pooling
    downsampled_tensor = F.avg_pool2d(image_tensor, kernel_size=kernel_size, stride=stride)

    # Convert the downsampled tensor back to a NumPy array
    downsampled_array = downsampled_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)

    return downsampled_array

# Plot clusters
def plot_image_cluster(image_array, cluster_centers_indices, labels, n_clusters_, alpha=0.25, marker_color = 'red', save = False, path = None, name = None):
  color_list = sns.color_palette(palette=cc.glasbey_category10,n_colors=n_clusters_)
  height, width, _ = image_array.shape

  label_matrix = labels.reshape((image_array.shape[0],image_array.shape[1]))
  cluster_center_coord = image_flatten[cluster_centers_indices,-2:]
  color_matrix = np.zeros((image_array.shape[0], image_array.shape[1], 3)) # Initialize with correct shape
  for i in range(n_clusters_):
    indices = np.where(label_matrix==i)
    color_matrix[indices] = color_list[i]
  plt.imshow(image_array[:,:,:3])
  plt.imshow(color_matrix,alpha = alpha)
  plt.scatter(cluster_center_coord[:,0]*width,cluster_center_coord[:,1]*height,marker='x',color=marker_color,s=100)
  if save:
    plt.axis('off')
    plt.savefig(path+name+".png",format='png',bbox_inches='tight')
    plt.savefig(path+name+".eps",format='eps',bbox_inches='tight')
    plt.axis('on')
  plt.show()

# Scale and downsample image_array into a 150*150 image
def scale_and_downsample(image_array, target_size=(150, 150)):
    """Scales and downsamples an image array to the target size.

    Args:
        image_array: A NumPy array representing the image.
        target_size: A tuple (width, height) specifying the target size.

    Returns:
        A NumPy array representing the scaled and downsampled image.
    """
    # Convert the NumPy array back to a PIL Image
    img = Image.fromarray((image_array * 255).astype(np.uint8))

    # Resize the image using Lanczos resampling for high-quality downsampling
    img = img.resize(target_size, Image.LANCZOS)

    # Convert the resized image back to a NumPy array
    resized_array = np.array(img) / 255.0

    return resized_array

def combine_images(images):
    """Combines multiple images into a single figure with indices.

    Args:
        images: A list of PIL Image objects.

    Returns:
        A matplotlib figure object.
    """

    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Adjust figsize as needed

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        ax = axes[row, col] if rows > 1 and cols > 1 else (axes[row] if cols == 1 else axes[col])
        try:
            # shrink img
            img = img.resize((100, 100), Image.LANCZOS)
            ax.imshow(img)
            ax.set_title(f"{i}", fontsize=10) # Adjusted font size
            ax.axis("off")
        except Exception as e:
          print(f"Error displaying image {i}: {e}")

    # Hide any empty subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 and cols > 1 else (axes[row] if cols == 1 else axes[col])
        ax.axis("off")

    # make layout even tighter
    plt.tight_layout(pad=0.1)  # Adjust pad as needed
    # plt.tight_layout()  # Adjust spacing between subplots
    return fig

"""# Data Setup"""

# Load JPEG Images into a list
image_dir = "banana"  # Replace with your image directory

images = load_jpeg_images(image_dir)

# Combine all images as one big figure with their index on it and plot
combined_figure = combine_images(images[:100])
plt.show()

image_index_list = [54]
figure_name = "banana3"
figure_path = "..." # Replace with your image output directory

for image_index in image_index_list:
    image_array = image_to_3d_array(images[image_index])  # Convert the first image
    if image_array is not None:
        print("Image shape:", image_array.shape)
        # Now you can work with image_array, a 3D NumPy array
        plt.imshow(image_array)
        plt.axis('off')
        # plt.savefig(figure_path+figure_name+".png",format='png',bbox_inches='tight')
        # plt.savefig(figure_path+figure_name+".eps",format='eps',bbox_inches='tight')
        plt.axis('on')
        plt.title(str(image_index))
        plt.show()

# Resize image
image_array = scale_and_downsample(image_array, target_size=(100, 100))
print("Resized image shape:", image_array.shape)
plt.imshow(image_array)
plt.show()

# Get the image dimensions
height, width, _ = image_array.shape

# Create x and y coordinate arrays
x_coords = np.arange(width)/(width-1)
y_coords = np.arange(height)/(height-1)

# Create meshgrids of the normalized coordinates
x_coords, y_coords = np.meshgrid(x_coords, y_coords)

# Reshape the coordinate arrays to match the image shape
x_coords = x_coords.reshape(-1, 1)
y_coords = y_coords.reshape(-1, 1)

# Concatenate the x and y coordinates along the last axis
coords = np.concatenate((x_coords, y_coords), axis=1)

# Reshape the coordinates to match the image array
coords = coords.reshape(height, width, 2)

# Append the normalized coordinates to the image array along the last axis
image_array = np.concatenate((image_array, coords), axis=2)

print("Image array with coordinates shape:", image_array.shape)

"""# Remove Correlated Sensors"""

# AffinityPropagation Clustering
image_flatten = image_array.reshape((image_array.shape[0]*image_array.shape[1],5)).astype("float16")  # (n_pixels X n_color_channels)
af = AffinityPropagation(damping=0.90,max_iter=10000, convergence_iter=5, copy=False, preference= -40, affinity='euclidean').fit(image_flatten)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(set(labels))
print(n_clusters_)

plot_image_cluster(image_array, cluster_centers_indices, labels, n_clusters_,alpha=0.7, marker_color = 'yellow', save = True, path = figure_path, name = figure_name+"_clustered")

sensors = cluster_centers_indices