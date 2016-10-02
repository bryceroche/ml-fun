from __future__ import division, print_function, absolute_import
import tflearn.datasets.mnist as mnist
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 0
img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

def bdd():

    X, Y, testX, testY = mnist.load_data(one_hot=True)

    Y_test = np_utils.to_categorical(testY, nb_classes)
    X_test = testX.reshape(testX.shape[0], img_rows, img_cols, 1)

    X_test = X_test.astype('float32')
    X_test /= 255

    model = ('TF_learn_MNIST.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

bdd()