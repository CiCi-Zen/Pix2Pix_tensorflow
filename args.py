from PIL import Image
# 👇 核心配置的参数

# 图片大小
image_shape = (64,64,3) # 64, 64, 3


# 每一轮 D 和 G 分别训练多少论
d_step = 6
g_step = 2 # 1


# G 和 D 的学习率
g_learning_rate = 0.00005
d_learning_rate = 0.0001 # 0.0001


# identity_loss weight，可以手调来试验
# 这个是identity_loss的权重，可以自己调，我试的是0.0001和0.001
id_weight = 0.0001 # 平衡生成器的生成质量和一致性
# Gradient Penalty的weight，用于惩罚 D 的权重更新，使 D 不要学得太快
gp_weight = 10


# 每轮训练数据批次大小
batch_size = 1

# 训练轮数
epochs = 50

# 训练数据的路径，
# 数据集路径如何写？
# 把所有图片装在一个文件夹，然后把地址写为它的上级文件夹
# 比如把所有图片放在D:/a/b/下，那么x_path写成D:/a。
# 数据集在这里下载，用迅雷复制链接下载比较快
# https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
# http://efrosgans.eecs.berkeley.edu/pix2pix/datasets
# 下载解压完成后按上述写路径就可以
# path = 'C:/Work/night2day/night2day/train'
# path = 'C:/Work/facades/facades/traini'
path = '/Users/zeno/Desktop/Pix2Pix-1/facades'



# 从训练集中提取输入 -> 根据你的数据集图片来更改下面的x_train和y_train，详细在note中介绍
def get_x_y_train(data): # data 为输入的数据
    # y_train 是右边的图像(提取了 data 中所有批次的右边部分的图像, image_shape[1] 表示单张图像的宽度, 切片操作 image_shape[1]:image_shape[1] * 2 表示从原图像的宽度一半的位置开始，直到宽度的两倍，这部分即为右边的图像)
    y_train = data[:, :, image_shape[1]:image_shape[1] * 2, :] # 训练输出的 y_train (返回张量)
    # x_train 是左边的图像
    x_train = data[:, :, 0:image_shape[1], :]  # 训练输出的 x_train (返回张量)

    return x_train,y_train



# 判断图像尺寸的方法
def imageSize(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        mode = img.mode  # 'RGB' 表示有3个通道
        print(f"👀 传入的图片尺寸 Image size: {width}x{height}, Mode: {mode}")




# 这里写你需要拿来测试的图片1和图片2的路径，比如你训练的是将结构图转成建筑物，那么
# 你可以选两张结构图
# test_image_path1 = 'C:/Work/night2day/night2day/train/train/1_50_to_15.jpg'
# test_image_path1 = 'C:/Work/facades/facades/traini/train/12.jpg'

# 👇 后续可以修改成轮询 ？
test_image_path1 = '/Users/zeno/Desktop/Pix2Pix-1/facades/train/12.jpg'

# test_image_path2 = 'C:/Work/night2day/night2day/train/train/1_83_to_109.jpg'
# test_image_path2 = 'C:/Work/facades/facades/traini/train/15.jpg'
test_image_path2 = '/Users/zeno/Desktop/Pix2Pix-1/facades/train/15.jpg'
