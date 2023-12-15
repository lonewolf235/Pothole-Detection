import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense

# Load the pre-trained VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the saved model weights
model.load_weights('model.h5')

# Define the size of the input images
img_size = (224, 224)

# Define the lower and upper bounds for the pothole detection threshold
lower_thresh = 0.5
upper_thresh = 0.9

# Create a video capture object to capture frames from the webcam
cap = cv2.VideoCapture('pothole_video.mp4')
width  = cap.get(3)
height = cap.get(4)
result = cv2.VideoWriter('results.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         25,(720,640))
# Loop over the video frames
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Resize the frame to the input size of the model
    frame = cv2.resize(frame, img_size)

    # Preprocess the frame for input to the model
    x = np.expand_dims(frame, axis=0)
    x = preprocess_input(x)

    # Pass the preprocessed frame through the model to predict whether it contains a pothole or not
    prediction = model.predict(x)

    # Check if the prediction exceeds the pothole detection threshold
    if prediction > lower_thresh and prediction < upper_thresh:
        # Draw a rectangle around the pothole
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)

    # Show the frame with the pothole detection overlay
    cv2.imshow('Pothole Detection', frame)
    result.write(frame)
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
