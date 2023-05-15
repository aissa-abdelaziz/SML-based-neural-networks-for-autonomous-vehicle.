from utlis import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Step 01 : Initialize Data
path = 'myData'
data = importDataInfo(path)

#Step 02 :  Visualize and Balance Data
data = balanceData(data,display=False)

#Step 3: Prepare for processing

imagesPath, steerings = loadData(path,data)
print(steerings)


#Step 4: Split for Training and Validation

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
# Split the 'imagesPath' and 'steerings' arrays into training and validation sets
# The 'test_size=0.2' parameter indicates that 20% of the data will be allocated for validation
# The 'random_state=10' parameter ensures reproducibility of the random shuffling
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#Step 5: Augmentation
#Step 6: Preprocessing

#Step 7: Training and Testing
model = createModel()
model.summary()


#Step 9: Training
history = model.fit(
    batchGen(xTrain, yTrain, 100, 1),  # Training data batch generator
    steps_per_epoch=200,              # Number of steps (batches) per epoch
    epochs=10,                        # Number of epochs to train
    validation_data=batchGen(xVal, yVal, 100, 0),  # Validation data batch generator
    validation_steps=200              # Number of steps (batches) for validation
)


#Step 10: Saving & Plotting

model.save('model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()