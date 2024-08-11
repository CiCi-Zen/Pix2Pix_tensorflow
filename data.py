from keras.preprocessing import image_dataset_from_directory
import args
import matplotlib.pyplot as plt
import tensorflow as tf

path = args.path



data = image_dataset_from_directory(directory=path,
                                    label_mode=None,
                                    batch_size=args.batch_size,
                                    image_size=(args.image_shape[0],args.image_shape[1]*2)
                                    # color_mode='grayscale'
                                   )
                                    #高  水平长度
if __name__ == '__main__':
    print(data)
    datagen = iter(data)
    x = next(datagen)
    # x = x/255. # 归一化, 将图像的像素值范围从[0, 255]缩放到[0, 1
    x = (x / 127.5) - 1 # 🔥 生成器模型使用了 tanh 作为最后一层的激活函数，因此输出的像素值范围在 [-1, 1] 之间。如果你的输入数据范围是 [0, 1]
    x = tf.squeeze(x,axis=0)
    plt.imshow(x)
    plt.show()

