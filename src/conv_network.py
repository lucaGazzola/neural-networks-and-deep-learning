import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

class CNNetwork(object):

    def run(self, k_size, p_size, d_size):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Reshaping the array to 4-dims so that it can work with the Keras API
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
        # Making sure that the values are float so that we can get decimal points after division
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print('Number of images in x_train', x_train.shape[0])
        print('Number of images in x_test', x_test.shape[0])

        model = Sequential()
        model.add(Conv2D(28, kernel_size=(k_size,k_size), input_shape=(28,28,1)))
        model.add(MaxPooling2D(pool_size=(p_size,p_size)))
        model.add(Flatten())
        model.add(Dense(d_size, activation=tf.nn.relu))
        model.add(Dense(10, activation=tf.nn.softmax))

        model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
        model.fit(x=x_train,y=y_train, epochs=5)

        score = model.evaluate(x_test, y_test)
        return score[1]
        