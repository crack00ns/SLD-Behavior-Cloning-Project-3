import csv
import cv2
import numpy as np
import pdb
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

# Data Path relative to your machine
DATA_PATH = "../data/"

# Hyperparameters
STEER_CORRECTION = 0.1  # Using a left and right image steer correction
BATCH_SIZE       = 128  
STEER_THRESHOLD  = 0.05 # 80% images with steering angle less than threshold
EPOCHS           = 15   # Maximum number of EPOCHS. (Early stopping)

# Data Preprocessing and Augmentation 
samples =[]
str_angle_orig = []
str_angle_aug = []
str_angle_drp = []

with open(DATA_PATH+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        centername = DATA_PATH+'IMG/'+row[0].split('/')[-1]
        leftname   = DATA_PATH+'IMG/'+row[1].split('/')[-1]
        rightname  = DATA_PATH+'IMG/'+row[2].split('/')[-1]
        
        steering_center = float(row[3])
        steering_left   = steering_center + STEER_CORRECTION
        steering_right  = steering_left   - STEER_CORRECTION
    
        str_angle_orig.append(steering_center)
  
        def append_samples(name, angle, flip):
            samples.append([name, angle, flip])
            str_angle_aug.append(angle)

        # Samples are created with each row containing, image name, steering and whether flipped image should be used.
        # Always add center image and its flipped version 
        append_samples(centername,steering_center, 0)
        append_samples(centername,-steering_center,1)

        if np.random.randint(3)!=0:  # Add the left and right correction with 2/3 probability to reduce dependency on correction parameters
            append_samples(leftname,  steering_left,  0)
            append_samples(rightname,-steering_right, 1)
            append_samples(leftname,  steering_left,  0)
            append_samples(rightname,-steering_right, 1)

samples = np.array(samples, dtype = object)

# Reduce data with low steering angles: Drop rows with probability 0.8
print("Number of samples before dropping low steering angles: {}".format(samples.shape[0]))
index = np.where((np.abs(samples[:,1])<STEER_THRESHOLD)==True)[0]
rows = [i for i in index if np.random.randint(10) < 9]
samples = np.delete(samples, rows, 0)
print("Removed %s rows with low steering"%(len(rows)))
print("Number of samples after dropping low steering angles: {}".format(samples.shape[0]))

for row in samples:
    str_angle_drp.append(row[1])

# Save histogram of steering angles
def save_hist(data, name):
    plt.figure()
    plt.hist(data, bins=20, color='green')
    plt.xlabel('Steering angles')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of '+name.replace("_"," "))
    plt.savefig(name+".png")

save_hist(str_angle_orig, "steering_angles_original")
save_hist(str_angle_aug,  "steering_angles_after_augmentation")
save_hist(str_angle_drp,  "steering_after_dropping_low_angles")

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Generator for Keras model.fit
def generator(samples, batch_size):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # print("Getting next batch")
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_name  = batch_sample[0]
                steering = batch_sample[1]
                flip     = batch_sample[2]
                img = cv2.imread(img_name) if flip==0 else cv2.flip(cv2.imread(img_name),1)  # Flip image if flip was 1 
                images.append(img)
                angles.append(steering)

            X_train = np.array(images)
            y_train = np.array(angles)
            # print("X_train.shape {}".format(X_train.shape))
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator      = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

# Keras model
model =  Sequential()
model.add(Lambda(lambda x : x / 255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) # trim image to only see section with road
model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
callbacks = [EarlyStopping(monitor='val_loss',patience=2,verbose=0)]
history_object = model.fit_generator(train_generator, steps_per_epoch = (len(train_samples)/BATCH_SIZE), validation_data=validation_generator, \
                validation_steps = (len(validation_samples)/BATCH_SIZE), epochs=EPOCHS, verbose = 1, callbacks=callbacks)
model.save("model.h5")

### plot the training and validation loss for each epoch
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')
