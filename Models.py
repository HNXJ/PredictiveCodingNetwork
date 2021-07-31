from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


class RNNModel1(tf.keras.Model):
    def __init__(self):
        super(RNNModel1, self).__init__()
        self.Input = (tf.keras.layers.InputLayer(input_shape=(None, 784)))
        self.LSTM = tf.keras.layers.LSTM(input_shape=(None, 784),
          units=512,
          recurrent_dropout=0.2,
          return_sequences=True,
          # return_state=True
        )
        self.FCN = tf.keras.layers.Dense(units=10)
        return 
        
    def call(self, x):
        out = self.Input(x)
        out = self.LSTM(x)
        out = self.FCN(out)
        return out

    def get_state(self):
        return
    
    
class PredictiveNet():
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(None, 784)))
        self.model.add(tf.keras.layers.LSTM(
          units=512,
          recurrent_dropout=0.2,
          return_sequences=True,
          # return_state=True
        ))
        self.model.add(tf.keras.layers.Dense(units=10))
        return

    def printw(self): # Debug log
        print(K.mean(self.model.layers[1].weights[0]))
        return

    def EnergyCostLoss(self, y_true, y_pred):
        error = y_pred - y_true
        lambda1 = 1
        lambda2 = 1
        return K.mean(K.square(error) + lambda1*K.mean(K.abs(y_pred))) + lambda2*K.mean(K.abs(self.model.layers[1].weights[0]))
    
    
