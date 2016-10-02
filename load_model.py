import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils




def abc():
    model = keras.models.load_model('my_model.h5')

    testdata1 = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
    #testdata2= np.array([[1,0],[0,1],[1,0],[1,1]], "float32")

    print model.predict(testdata1).round()
    print model.predict(testdata1)

    text_file = open("model_architecture.txt", "w")
    text_file.write(model.to_json())
    text_file.close()
