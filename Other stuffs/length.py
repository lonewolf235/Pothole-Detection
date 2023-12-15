import os
import shutil
from sklearn.model_selection import train_test_split


# Set the path to the directory containing the images
img_dir = 'D:/pothole_dataset/pothole_image_data'

# Define the paths to the training and testing directories
train_dir = 'D:\pothole_dataset/train'
test_dir = 'D:\pothole_dataset/test'

# Define the percentage of images to use for testing
test_size = 0.2

# Get a list of the image filenames in the directory
image_filenames = os.listdir(img_dir)
print(len(image_filenames))
# Split the image filenames into training and testing sets
train_filenames, test_filenames = train_test_split(image_filenames, test_size=test_size)

# Create the training directory if it doesn't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

# Create the testing directory if it doesn't exist
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Move the training images to the training directory
for filename in train_filenames:
    src_path = os.path.join(img_dir, filename)
    dst_path = os.path.join(train_dir, filename)
    shutil.copy(src_path, dst_path)

# Move the testing images to the testing directory
for filename in test_filenames:
    src_path = os.path.join(img_dir, filename)
    dst_path = os.path.join(test_dir, filename)
    shutil.copy(src_path, dst_path)