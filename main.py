from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


from DataPreparation import *
from Models import *


# GPU config if needed
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
print('Found GPU at: {}'.format(device_name))


X, Y, Xn, Yn = init_MNIST()
Net1 = PredictiveNet()
# a = model.predict(Xn[1:2, :, :])
Net1.model.compile(
  loss=Net1.EnergyCostLoss,
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.003)
)
Net1.model.summary()

Net1.printw()

L1 = K.function([Net1.model.layers[0].input], [Net1.model.layers[1].output])

history = Net1.model.fit(
    x=X, y=Y,
    epochs=50,
    batch_size=5,
    validation_split=0.0,
    verbose=1,
    shuffle=True
)

means1 = []
for k in range(10):
    l, r = k, k+1
    l_pred = L1(Xn[l:r, :, :])[0]
    y_pred = Net1.model.predict(Xn[l:r, :, :])

    # for i in range(7):
        # plt.plot(l_pred[0][0, :, i])
    # for i in range(10):
    # plt.plot(y_pred[0, :, 6])
    # for i in range(0, 100, 10):
    #     print(Yn[l:r, i, :])
    means1.append(np.mean(np.mean(np.abs(l_pred))))
    
means2 = []
for k in range(20, 40):
    l, r = k, k+1
    l_pred = L1(Xn[l:r, :, :])[0]
    y_pred = Net1.model.predict(Xn[l:r, :, :])

    # for i in range(7):
    #     plt.plot(l_pred[0, i, :])
    # for i in range(10):
    # plt.plot(y_pred[0, :, 6])
    # for i in range(0, 100, 10):
    #     print(Yn[l:r, i, :])
    means2.append(np.mean(np.mean(np.abs(l_pred))))
    
print(np.mean(means1), np.mean(means2))