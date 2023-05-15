import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam
import cv2


def getName(filePath):
    return filePath.split('\\')[-1] #we split the path and return only the last part


def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed'] #we add the names to the columns of our excel file
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns) #we read the path from the excel file
    ## REMOVE FILE PATH AND GET ONLY FILE NAME
    print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName) #we return the name associated with the images obtained from the centered sensor
    print(data.head())
    print('Total Images Imported', data.shape[0])
    return data #return the data informations




def balanceData(data,display=True):
    nBin = 31
    samplesPerBin = 500
    # Compute the histogram of the 'Steering' data
    hist, bins = np.histogram(data['Steering'], nBin)
    if display:
        # Compute the center of each bin
        center = (bins[:-1] + bins[1:]) * 0.5
        # Plot the histogram
        plt.bar(center, hist, width=0.06)
        # Plot a line indicating the desired number of samples per bin
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        # Display the plot
        plt.show()

    removeindexList = []

    # Iterate over each bin
    for j in range(nBin):
        binDataList = []
        # Iterate over each data point
        for i in range(len(data['Steering'])):
            # Check if the data point falls within the current bin
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)

        # Shuffle the data points within the bin
        binDataList = shuffle(binDataList)
        # Remove excess data points from the bin
        binDataList = binDataList[samplesPerBin:]
        # Extend the list of indices to be removed
        removeindexList.extend(binDataList)


    # Print the number of removed images
    print('Removed Images:', len(removeindexList))

    # Remove the corresponding rows from the 'data' DataFrame
    data.drop(data.index[removeindexList], inplace=True)

    # Print the number of remaining images
    print('Remaining Images:', len(data))

    if display:
        # Compute the histogram of the 'Steering' data after balancing
        hist, _ = np.histogram(data['Steering'], (nBin))

        # Plot the balanced histogram
        plt.bar(center, hist, width=0.06)

        # Plot a line indicating the desired number of samples per bin
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))

        # Display the plot
        plt.show()

    return data



def loadData(path, data):
  imagesPath = []
  steering = []

  # Iterate over each row in the 'data' DataFrame
  for i in range(len(data)):
    indexed_data = data.iloc[i]

    # Extract the image path from the first column of the row
    imagesPath.append(f'{path}/IMG/{indexed_data[0]}')

    # Extract the steering angle from the fourth column of the row
    steering.append(float(indexed_data[3]))

# Convert the lists to NumPy arrays
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)

  # Return the arrays
  return imagesPath, steering




def augmentImage(imgPath,steering):
    img = mpimg.imread(imgPath)

    # Data augmentation techniques
    # Pan augmentation
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    # Zoom augmentation
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    #Brightness augmentation
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    #Flip augmentation
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    # Return the augmented image and steering angle
    return img, steering



def preProcess(img):
    # Crop the image to keep the region of interest
    img = img[60:135,:,:]
    # Convert the color space from RGB to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    # Resize the image to the desired dimensions
    img = cv2.resize(img, (200, 66))
    # Normalize the image by scaling pixel values to the range [0, 1]
    img = img/255
    # Return the preprocessed image
    return img


def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        # Generate a batch of images and steering angles
        for i in range(batchSize):
            # Randomly select an index within the range of available images
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                # Apply data augmentation if 'trainFlag' is True
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                # Read the image without augmentation if 'trainFlag' is False
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            # Append the preprocessed image and steering angle to the batch lists
            imgBatch.append(img)
            steeringBatch.append(steering)

        # Yield the batch as a tuple of NumPy arrays
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))



def createModel():
    # Create a Sequential model
    model = Sequential()
    # Add convolutional layers
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    # Flatten the output of the last convolutional layer
    model.add(Flatten())
    # Add fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    # Add the output layer with a single neuron (regression task)
    model.add(Dense(1))

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(Adam(learning_rate=0.0001), loss='mse')

    # Return the model
    return model