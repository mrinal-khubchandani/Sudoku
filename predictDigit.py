import tensorflow as tf
import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
import cv2
import matplotlib.pyplot as plt
import numpy as np

def classify(img):
    img1 = img.copy()
    img1 = img1.astype('float32')
    img1 /= 255
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28))
    img1 = img1.reshape(1,28,28,1)
    model = get_model()
    y_hat = model.predict(img1)
    return np.argmax(y_hat)

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    batch_size = 32
    epochs = 4

    try:
        model.load_weights('./model_weights')
    except:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
        X_train = np.array(X_train, dtype=np.float64)
        X_test = np.array(X_test, dtype=np.float64)
        X_train /= 255
        X_test /= 255
        model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test, y_test))
        model.save_weights('./model_weights')
    
    return model