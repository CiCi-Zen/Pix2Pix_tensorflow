import tensorflow as tf
# 🔥要使用 python 来运行而不是 python3 !!!
# 👉 python testGPU.py !!!!!
if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Num GPUs:", len (physical_devices))
    print("Tensorflow Version:", tf.__version__)


# import numpy as np
# import tensorflow as tf
 
# x = np.random.random((10000, 5))
# y = np.random.random((10000, 2))
# x2 = np.random.random((2000, 5))
# y2 = np.random.random((2000, 2))
 
# inp = tf.keras.layers.Input(shape = (5,))
# l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(inp)
# l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
# l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
# l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
# l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
# o = tf.keras.layers.Dense(2, activation = 'sigmoid')(l1)
# model = tf.keras.models.Model(inputs = [inp], outputs = [o])
# model.compile(optimizer = "Adam", loss = "mse")