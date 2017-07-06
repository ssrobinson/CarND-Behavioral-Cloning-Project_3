import numpy as np
import keras
import matplotlib.pyplot as plt
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *

### variables for resized images used to train model ###
img_height = 16
img_width = 32
### batch size and epochs ###
batch_size = 128
nb_epoch = 10
### folder filepath for log data and images ###
data_filepath = '/Users/srobinson/CarND-Term1-Starter-Kit/data/'


### preproccesing training data to keep only S channel in HSV color space, and resize to 16 X 32 ###
def preprocess(img):
	#new_image = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[:],(img_cols,img_rows))
	#return new_image
	new_image = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(img_height,img_width))
	return new_image

### load training data ###
def load_data(X,y,data_filepath,steering_delta=0.4):

	log_path = '/Users/srobinson/CarND-Term1-Starter-Kit/data/driving_log.csv'
	logs = []

### import data log files ###
	with open(log_path,'rt') as f:
		reader = csv.reader(f)
		for line in reader:
			logs.append(line)
		log_labels = logs.pop(0)

### load center image ###
	for i in range(len(logs)):
		img_path = logs[i][0]
		img_path = data_filepath + 'IMG' + (img_path.split('IMG')[1]).strip()
		img = plt.imread(img_path)
		X.append(preprocess(img))
		y.append(float(logs[i][3]))

### load left image ###
	for i in range(len(logs)):
		img_path = logs[i][1]
		img_path = data_filepath + 'IMG' + (img_path.split('IMG')[1]).strip()
		img = plt.imread(img_path)
		X.append(preprocess(img))
		y.append(float(logs[i][3]) + steering_delta)

### load right image ###
	for i in range(len(logs)):
		img_path = logs[i][2]
		img_path = data_filepath + 'IMG' + (img_path.split('IMG')[1]).strip()
		img = plt.imread(img_path)
		X.append(preprocess(img))
		y.append(float(logs[i][3]) - steering_delta)


if __name__ == '__main__':

### load data for training ###

	print('Loading images and steering angles...')

	data={}
	data['features'] = []
	data['labels'] = []

	load_data(data['features'], data['labels'], data_filepath, 0.4)

	X_train = np.array(data['features']).astype('float32')
	y_train = np.array(data['labels']).astype('float32')

### Flip images about the y-axis, reverse cooresponding steering angle direction, and add to the original image data ###
	X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
	y_train = np.append(y_train,-y_train,axis=0)

### Split training and validation data ###
### Validation set is 10% of total dataset ###
	X_train, y_train = shuffle(X_train, y_train)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

### reshape new images to make sure all images have correct dimension ###
	X_train = X_train.reshape(X_train.shape[0], img_height, img_width, 1)
	X_val = X_val.reshape(X_val.shape[0], img_height, img_width, 1)


### Create model using Keras ###
	print('Creating model architecture...')

### Nvidia Model ###
### http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf ###

'''def nvidia_model(img_height=16, img_width=32, img_channels=3,
                       dropout=.4):

    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape), name='Normalization'))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0], kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout
    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))
        model.add(Dropout(dropout))

    # logit output - steering angle
    model.add(Dense(1, activation='elu', name='Out'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')
    return model'''

### Commaai model ###
### https://github.com/commaai/research/blob/master/train_steering_model.py ###
'''ch, row, col = 3, 16, 32  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col),output_shape=(ch, row, col)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model'''

### Custom Model ###
model = Sequential([
			Lambda(lambda x: x/127.5 - 1.,input_shape=(img_height,img_width,1)),
			Conv2D(2, 3, 3, border_mode='valid', input_shape=(img_height,img_width,1), activation='relu'),
			MaxPooling2D((4,4),(4,4),'valid'),
			Dropout(0.3),
			Flatten(),
			Dense(1)])
model.summary()

### Training model ###
print('Training model...')

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data = (X_val, y_val))

### Save model and weights ###
print('Saving model...')

model_json = model.to_json()
with open('model.json', 'w') as json_file:
		json_file.write(model_json)
model.save_weights('model.h5')
print('Model Saved.')
