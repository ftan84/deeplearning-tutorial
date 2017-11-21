import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
K.set_image_dim_ordering('th')

seed = 120
# Add comment here
# Another comment here
np.random.seed(seed)
# How about writing a whole
# long comment here?
# another comment line

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
x_train = x_train / 255
x_test = x_test / 255
# comment

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    '''Baseline model using mlp'''
    model = Sequential()
    model.add(Dense(num_pixels,
                    input_dim=num_pixels,
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(num_classes,
                    kernel_initializer='normal',
                    activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=10,
          batch_size=200,
          verbose=1)
scores = model.evaluate(x_test, y_test, verbose=0)
print('Baseline error: %.2f%%' % (100 - scores[1] * 100))




seed = 120
np.random.seed(seed)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
x_train = x_train / 255
x_test = x_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def cnn_baseline():
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=(1, 28, 28),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = cnn_baseline()
model.summary()

model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=100,
          batch_size=128,
          verbose=1)
scores = model.evaluate(x_test, y_test, verbose=0)
print('Baseline CNN: %.2f%%' % (100 - scores[1] * 100))

