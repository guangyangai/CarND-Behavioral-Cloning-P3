import os
import csv
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split

CORRECTION = 0.2

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				center_image_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
				left_image_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
				right_image_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
				center_image_original = cv2.imread(center_image_name)
				left_image_original = cv2.imread(left_image_name)
				right_image_original = cv2.imread(right_image_name)
				#convert from BGR to RGB
				center_image = cv2.cvtColor(center_image_original,cv2.COLOR_BGR2RGB)
				left_image = cv2.cvtColor(left_image_original,cv2.COLOR_BGR2RGB)
				right_image = cv2.cvtColor(right_image_original,cv2.COLOR_BGR2RGB)
				center_angle = float(batch_sample[3])
				left_angle = center_angle + CORRECTION
				right_angle = center_angle - CORRECTION
				images.extend([center_image, left_image, right_image])
				angles.extend([center_angle, left_angle, right_angle])
			#augment image
			augmented_images, augmented_angles = [], []
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_images.append(cv2.flip(image,1))
				augmented_angles.append(angle)
				augmented_angles.append(angle*-1.0)
			# trim image to only see section with road
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

samples = []
training_file = 'data/driving_log.csv'
with open(training_file, 'r') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for line in reader:
		samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#build model
model = Sequential()
#cropping
model.add(Cropping2D(cropping=((70,25), (0,0))))
#normalizing
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
#network
model.add(Convolution2D(6,5,5, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Convolution2D(16,5,5, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')


#use callback to save history
filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history_object = model.fit_generator(train_generator, samples_per_epoch =
	len(train_samples), validation_data = 
	validation_generator,
	nb_val_samples = len(validation_samples), 
	nb_epoch=10, verbose=1, callbacks=[model_checkpoint])

model.save('model.h5')
