from generator import build_generator
from discriminator import build_discriminator
from pix2pix import pix2pix
from callbacks import Callback
from data import data
import args
import tensorflow as tf
from PIL import Image

# 【核心运行此脚本】!!!!
# 🔥要使用 python 来运行而不是 python3 !!!
# 👉 python testGPU.py !!!!!

# 判别器损失
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

# 生成器损失
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


if __name__ == '__main__':
    generator = build_generator()
    discriminator = build_discriminator()
    cbk = Callback()
    
    # 实例化 pix2pix 类
    pix2pix = pix2pix(generator,discriminator)
    pix2pix.compile(gen_optimizer=tf.keras.optimizers.Adam(learning_rate=args.g_learning_rate,beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                    g_loss_fn=generator_loss,
                    d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.d_learning_rate,beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                    d_loss_fn=discriminator_loss)
    pix2pix.fit(data,batch_size=args.batch_size,epochs=args.epochs,callbacks=[cbk])
    generator.save('generator.h5')
    discriminator.save('discriminator.h5')