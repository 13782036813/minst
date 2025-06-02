from tensorflow import keras
from keras import layers
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np

model = keras.Sequential([
    layers.Dense(units=784, activation='relu'),
    layers.Dense(units=256, activation='relu'),
    layers.Dense(units=10, activation='softmax'),
])

model.name = 'hch_model'



def get_data(train_size, test_size):
    # train_data
    x_train = np.zeros((train_size, 784))
    y_train = np.zeros((train_size, 10))
    labels = np.loadtxt('./train_labs.txt', dtype=int)
    for i in range(train_size):
        img = plt.imread(f'./train/{i}.png').reshape(784,) / 255.0  
        x_train[i] = img
        if i % 100 == 0:
            print(f'Processing training images: {i}')
        label = labels[i, 1]
        y_train[i, label] = 1.0

    # test_data
    x_test = np.zeros((test_size, 784))
    y_test = np.zeros((test_size, 10))
    labels = np.loadtxt('./test_labs.txt', dtype=int)
    for i in range(test_size):
        img = plt.imread(f'./test/{i}.png').reshape(784,) / 255.0  
        x_test[i] = img
        if i % 100 == 0:
            print(f'Processing test images: {i}')
        label = labels[i, 1]
        y_test[i, label] = 1.0
    
    return x_train, y_train, (x_test, y_test)


x_train, y_train, test_datas = get_data(8000,1000)

model.compile(optimizer='adam',
              loss=categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=test_datas)
model.save('hch_model.h5')

model.summary()

