import tensorflow as tf
from keras import layers
import args


# 判别器
def build_discriminator():
    input1 = layers.Input(shape=args.image_shape)
    input2 = layers.Input(shape=args.image_shape)
    model_input = layers.Concatenate(axis=-1)([input1, input2])
    
    # 👇 经过卷积层
    x = layers.Conv2D(128,2,1,padding='same')(model_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256,2,1,padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256,2,1,padding='same')(x)
    x = layers.LeakyReLU()(x)
    output = layers.Conv2D(1,3,3,activation='tanh')(x)

    discriminator = tf.keras.Model([input1,input2],output,name='discriminator')

    return discriminator # 输出 N X N 的矩阵

if __name__ == '__main__':
    discriminator = build_discriminator()
    discriminator.summary()