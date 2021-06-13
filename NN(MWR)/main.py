# baseline cnn model for mnist
from numpy import mean
from numpy import std
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))
	testX = testX.reshape((testX.shape[0], 32, 32, 3))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def model_one():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_two():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_three():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation
def fit_model(dataX,dataY):
	model = model_one()
	model1 = model_two()
	model2 = model_three()
	model.fit(dataX, dataY, epochs=10, batch_size=32, validation_data=(dataX, dataY))
	model1.fit(dataX, dataY, epochs=10, batch_size=32, validation_data=(dataX, dataY))
	model2.fit(dataX, dataY, epochs=10, batch_size=32, validation_data=(dataX, dataY))
	return model,model1,model2


def evaluate_model(dataX, dataY, model):
	_, acc = model.evaluate(dataX, dataY)
	print('> %.3f' % (acc * 100.0))
	return acc


def run_test_harness():

	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	trainX = trainX[0:3000]
	trainY = trainY[0:3000]
	testY = testY[0:100]
	testX = testX[0:100]

	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	#evaluate model
	model, model1, model2 = fit_model(trainX, trainY)
	evaluate_model(testX, testY,model)
	evaluate_model(testX, testY,model1)
	evaluate_model(testX, testY,model2)

	# scores, histories = evaluate_model(trainX, trainY,model)
	for i in range(10):
		value = model.predict(testX[i:i+1])
		tmp = np.argmax(value[0])
		pyplot.imshow(testX[i], cmap=pyplot.get_cmap('gray'))
		pyplot.suptitle(f' Model 1,Predicted:  {tmp}, correct : {np.argmax(testY[i])}')
		pyplot.show()
		# learning curve
	for i in range(10):
		value = model1.predict(testX[i:i + 1])
		tmp = np.argmax(value[0])
		pyplot.imshow(testX[i], cmap=pyplot.get_cmap('gray'))
		pyplot.suptitle(f' Model 2,Predicted:  {tmp}, correct : {np.argmax(testY[i])}')
		pyplot.show()
	for i in range(10):
		value = model2.predict(testX[i:i + 1])
		tmp = np.argmax(value[0])
		pyplot.imshow(testX[i], cmap=pyplot.get_cmap('gray'))
		pyplot.suptitle(f' Model 3,Predicted:  {tmp}, correct : {np.argmax(testY[i])}')
		pyplot.show()

# entry point, run the test harness
run_test_harness()