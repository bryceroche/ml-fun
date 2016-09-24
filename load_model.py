import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.models import load_model


model = keras.models.load_model('my_model.h5')

testdata1 = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
#testdata2= np.array([[1,0],[0,1],[1,0],[1,1]], "float32")

print model.predict(testdata1).round()
print model.predict(testdata1)

