import os
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Set the path to the directory containing the images
img_dir = 'D:/pothole_dataset/pothole_image_data'

# Define the paths to the training and testing directories
train_dir = 'D:\pothole_dataset/train'
test_dir = 'D:\pothole_dataset/test'

# Define the size of the input images
img_size = (224, 224)

# Create an instance of the VGG16 pre-trained model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze the layers of the pre-trained model so they are not updated during training
for layer in vgg16.layers:
    layer.trainable = False

# Define a new model to classify the potholes based on the features learned by VGG16
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Define data generators to augment the training data and preprocess the validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Define the training and validation data directories
train_dir = train_dir
val_dir = test_dir

# Use the data generators to load the images from the directories and preprocess them
train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=32, class_mode='binary')

# Train the model using the generators
model.fit_generator(train_generator, steps_per_epoch=train_generator.n // train_generator.batch_size, epochs=10,
                    validation_data=val_generator, validation_steps=val_generator.n // val_generator.batch_size)

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=32, class_mode='binary')
score = model.evaluate_generator(test_generator, steps=test_generator.n // test_generator.batch_size, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('model.h5')