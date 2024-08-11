import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import array_to_img
import args


## callback 写的主要是：在每一轮训练的末尾选择两个图像，把G生成的图像画出来，并保存下来
class Callback(keras.callbacks.Callback):
    def processing_image(self,image_path):
        #写这个函数主要的目的是读取图片和预处理
        image = tf.keras.preprocessing.image.load_img(image_path,target_size=(args.image_shape[0],args.image_shape[1] * 2))
        # ,color_mode='grayscale'
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.
        image = tf.expand_dims(image,axis=0)
        return image

    def __init__(self,num_figs=6):
        self.num_figures = num_figs

    # 👇 每运行一次就会画出图片结果进行比对
    def on_epoch_end(self, epoch, logs=None):
        # sample 1 第一张照片，包含了预处理函数和转换成矩阵函数，比较标准的操作
        test_image_1 = self.processing_image(args.test_image_path1) # 每次生成的图片结果 与 test_image_path1 进行比对
        facade = test_image_1[:, :, args.image_shape[1]:args.image_shape[1] * 2, :]
        ground_truth = test_image_1[:, :, 0:args.image_shape[1], :]
        gen_image = self.model.generator.predict(facade)
        gen_image = array_to_img(tf.squeeze(gen_image, axis=0))
        facade = array_to_img(tf.squeeze(facade, axis=0) / 255.)
        ground_truth = array_to_img(tf.squeeze(ground_truth, axis=0))

        # sample 2 第二张图片，同上
        test_image_2 = self.processing_image(args.test_image_path2) # 每次生成的图片结果 与 test_image_path2 进行比对
        ground_truth2 = test_image_2[:, :, 0:args.image_shape[1], :]
        facade2 = test_image_2[:, :, args.image_shape[1]:args.image_shape[1] * 2, :]
        gen_image2 = self.model.generator.predict(facade2)
        gen_image2 = array_to_img(tf.squeeze(gen_image2, axis=0))
        facade2 = array_to_img(tf.squeeze(facade2, axis=0))
        ground_truth2 = array_to_img(tf.squeeze(ground_truth2, axis=0))


        # 下面是绘图部分，也是比较固定的写法
        plot_img = [facade, gen_image, ground_truth,
                    facade2, gen_image2, ground_truth2]

        plot_title = ['facade', 'generated', 'ground_truth',
                      'facade', 'generated', 'ground_truth']

        plt.figure(figsize=(20, 20))
        for i in range(0, 6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(plot_img[i])
            # , cmap = 'gray'
            plt.title(plot_title[i])
            plt.axis('off')

        plt.savefig("epc_{ep}.png".format(ep=epoch+1))
        plt.close()


